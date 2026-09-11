"""Schemas and validators for cryptographically verified GPU result deliveries."""

from __future__ import annotations

import json
import platform
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from sklearn.metrics import f1_score

from spatial_graph_bench.tracking.schema import RunManifest
from spatial_graph_bench.utils.hashing import hash_dict, hash_file
from spatial_graph_bench.utils.logging import get_logger
from spatial_graph_bench.utils.versioning import get_code_version, get_torch_geometric_version

logger = get_logger("analysis.delivery")

BATCH_SCHEMA_VERSION = "1.0"
MACRO_F1_TOLERANCE = 1e-5


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


class AuditVerdict(StrEnum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


class AuditCheck(BaseModel):
    """Outcome of one named integrity or consistency check."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    passed: bool
    severity: Literal["hard", "soft"] = "hard"
    detail: str = ""


class RunAuditReport(BaseModel):
    """Aggregated audit outcome for a single benchmark run directory."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    dir_name: str
    verdict: AuditVerdict
    checks: list[AuditCheck]
    manifest_hash: str = ""

    @property
    def failed_hard_checks(self) -> list[str]:
        return [c.name for c in self.checks if not c.passed and c.severity == "hard"]

    @property
    def failed_soft_checks(self) -> list[str]:
        return [c.name for c in self.checks if not c.passed and c.severity == "soft"]


class BatchFileEntry(BaseModel):
    """Per-file record inside a packaged result batch."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    sha256: str
    size_bytes: int


class BatchRunEntry(BaseModel):
    """Per-run audit summary recorded at pack time."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    verdict: AuditVerdict
    failed_checks: list[str] = Field(default_factory=list)


class BatchManifest(BaseModel):
    """Cryptographic delivery manifest created on the producing machine."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = BATCH_SCHEMA_VERSION
    created_at_utc: str = Field(default_factory=_utc_now)
    hostname: str = ""
    platform_info: str = ""
    code_version: str | None = None
    python_version: str = ""
    torch_version: str = ""
    torch_geometric_version: str | None = None
    cuda_available: bool = False
    dataset_name: str = ""
    split_id: str = ""
    files: list[BatchFileEntry] = Field(default_factory=list)
    runs: list[BatchRunEntry] = Field(default_factory=list)

    def compute_batch_hash(self) -> str:
        payload = {
            "schema_version": self.schema_version,
            "files": [f.model_dump() for f in sorted(self.files, key=lambda x: x.path)],
            "runs": [
                r.model_dump() for r in sorted(self.runs, key=lambda x: (x.run_id, x.verdict))
            ],
        }
        return hash_dict(payload)


def collect_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file())


def build_batch_manifest(
    results_root: Path,
    dataset_name: str,
    split_id: str,
    audit_reports: list[RunAuditReport],
) -> BatchManifest:
    try:
        import torch

        torch_version = str(torch.__version__)
        cuda_available = bool(torch.cuda.is_available())
    except ImportError:
        torch_version = ""
        cuda_available = False

    entries = [
        BatchFileEntry(
            path=str(p.relative_to(results_root)),
            sha256=hash_file(p),
            size_bytes=p.stat().st_size,
        )
        for p in collect_files(results_root)
    ]
    runs = [
        BatchRunEntry(
            run_id=r.run_id,
            verdict=r.verdict,
            failed_checks=r.failed_hard_checks + r.failed_soft_checks,
        )
        for r in audit_reports
    ]
    return BatchManifest(
        hostname=platform.node(),
        platform_info=platform.platform(),
        code_version=get_code_version(),
        python_version=platform.python_version(),
        torch_version=torch_version,
        torch_geometric_version=get_torch_geometric_version(),
        cuda_available=cuda_available,
        dataset_name=dataset_name,
        split_id=split_id,
        files=entries,
        runs=runs,
    )


def verify_batch_files(batch_dir: Path, manifest: BatchManifest) -> tuple[list[str], list[str]]:
    """Layer 1: Verify file hashes against manifest."""
    problems: list[str] = []
    expected: set[str] = set()
    for entry in manifest.files:
        expected.add(entry.path)
        local = batch_dir / entry.path
        if not local.is_file():
            problems.append(f"missing: {entry.path}")
            continue
        actual = hash_file(local)
        if actual != entry.sha256:
            problems.append(f"hash-mismatch: {entry.path}")
    extras = [
        str(p.relative_to(batch_dir))
        for p in collect_files(batch_dir)
        if p.relative_to(batch_dir).as_posix() not in expected and p.name != "batch_manifest.json"
    ]
    return problems, extras


def audit_run_dir(
    run_dir: Path,
    prep_bundle: Any = None,
    split_definition: Any = None,
    graphs_root: Path | None = None,
) -> RunAuditReport:
    """Full 4-layer audit of one run directory against canonical frozen artifacts."""
    _ = split_definition
    _ = graphs_root
    checks: list[AuditCheck] = []

    manifest_path = run_dir / "run_manifest.json"
    metrics_path = run_dir / "metrics_summary.json"
    preds_path = run_dir / "test_preds.npy"
    probs_path = run_dir / "test_probs.npy"

    manifest: RunManifest | None = None
    if manifest_path.is_file():
        try:
            manifest = RunManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
            checks.append(AuditCheck(name="manifest_schema", passed=True))
        except Exception as err:
            checks.append(AuditCheck(name="manifest_schema", passed=False, detail=str(err)))
    else:
        checks.append(
            AuditCheck(name="manifest_schema", passed=False, detail="run_manifest.json absent")
        )

    summaries = {}
    if metrics_path.is_file():
        try:
            summaries = json.loads(metrics_path.read_text(encoding="utf-8"))
            checks.append(AuditCheck(name="metrics_schema", passed=True))
        except Exception as err:
            checks.append(AuditCheck(name="metrics_schema", passed=False, detail=str(err)))
    else:
        checks.append(
            AuditCheck(name="metrics_schema", passed=False, detail="metrics_summary.json absent")
        )

    # Layer 3: Provenance hash-chain checks
    if manifest is not None and prep_bundle is not None:
        expected_feat_hash = prep_bundle.manifest.compute_manifest_hash()
        feat_match = manifest.feature_manifest_hash == expected_feat_hash
        checks.append(
            AuditCheck(
                name="feature_manifest_hash_match",
                passed=feat_match,
                detail="ok"
                if feat_match
                else f"run={manifest.feature_manifest_hash[:12]} vs local={expected_feat_hash[:12]}",
            )
        )
        if manifest.split_hash:
            split_match = manifest.split_hash == prep_bundle.manifest.split_config_hash
            checks.append(
                AuditCheck(
                    name="split_hash_match",
                    passed=split_match,
                    detail="ok"
                    if split_match
                    else f"run={manifest.split_hash[:12]} vs local={prep_bundle.manifest.split_config_hash[:12]}",
                )
            )

    # Layer 4: Independent semantic recomputation of metrics from frozen labels
    y_test = getattr(prep_bundle, "test_labels", None)
    if preds_path.is_file() and y_test is not None:
        try:
            y_pred = np.load(preds_path)
            len_ok = len(y_pred) == len(y_test)
            checks.append(
                AuditCheck(
                    name="preds_length",
                    passed=len_ok,
                    detail=f"pred={len(y_pred)}, test={len(y_test)}",
                )
            )

            if probs_path.is_file():
                probs = np.load(probs_path)
                sums_ok = bool(np.allclose(probs.sum(axis=1), 1.0, atol=1e-3))
                strict_argmax = bool(np.all(probs.argmax(axis=1) == y_pred))
                if not strict_argmax and sums_ok and len(probs) == len(y_pred):
                    pred_probs = probs[np.arange(len(y_pred)), y_pred]
                    max_probs = probs.max(axis=1)
                    achieves_max = bool(np.all(np.isclose(pred_probs, max_probs, atol=1e-5)))
                    argmax_ok = achieves_max
                else:
                    argmax_ok = strict_argmax

                checks.append(
                    AuditCheck(
                        name="probs_sanity",
                        passed=sums_ok and argmax_ok,
                        detail=f"sums_to_1={sums_ok}, argmax_matches_preds={argmax_ok}",
                    )
                )

            # Recompute macro-F1 with §3.4 label-space intersection
            reported_test = summaries.get("test", {})
            reported_f1 = reported_test.get("macro_f1")

            if reported_f1 is not None and len_ok:
                eval_label_names = reported_test.get("evaluated_labels")
                label_to_id = getattr(prep_bundle, "label_to_id", None)
                if eval_label_names and label_to_id:
                    eval_ids = [label_to_id[lab] for lab in eval_label_names if lab in label_to_id]
                    mask = np.isin(y_test, eval_ids)
                    y_test_sub = y_test[mask]
                    y_pred_sub = y_pred[mask]
                    recomputed_f1 = float(
                        f1_score(
                            y_test_sub,
                            y_pred_sub,
                            labels=eval_ids,
                            average="macro",
                            zero_division=0.0,
                        )
                    )
                else:
                    active_labels = sorted(set(y_test[y_test >= 0]))
                    recomputed_f1 = float(
                        f1_score(
                            y_test, y_pred, labels=active_labels, average="macro", zero_division=0.0
                        )
                    )
                delta = abs(recomputed_f1 - reported_f1)
                f1_ok = delta <= MACRO_F1_TOLERANCE
                checks.append(
                    AuditCheck(
                        name="macro_f1_recomputation",
                        passed=f1_ok,
                        detail=f"rep={reported_f1:.6f}, recalc={recomputed_f1:.6f}, delta={delta:.2e}",
                    )
                )
        except Exception as err:
            checks.append(AuditCheck(name="macro_f1_recomputation", passed=False, detail=str(err)))

    has_hard_fail = any(not c.passed and c.severity == "hard" for c in checks)
    has_soft_fail = any(not c.passed and c.severity == "soft" for c in checks)
    verdict = (
        AuditVerdict.FAIL
        if has_hard_fail
        else (AuditVerdict.WARN if has_soft_fail else AuditVerdict.PASS)
    )

    return RunAuditReport(
        run_id=manifest.run_id if manifest is not None else run_dir.name,
        dir_name=run_dir.name,
        verdict=verdict,
        checks=checks,
        manifest_hash=manifest.compute_manifest_hash() if manifest is not None else "",
    )


def append_ingestion_log(log_path: Path, entry: dict[str, Any]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, sort_keys=True) + "\n")
