"""Graph bundle container, PyG converter, and topological manifest."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field
from torch_geometric.data import Data

from spatial_graph_bench.utils.hashing import hash_dict


class GraphManifest(BaseModel):
    """Cryptographic and topological manifest for a serialized graph bundle."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    graph_name: str
    builder_type: str
    dataset_name: str
    split_id: str
    protocol_variant: str = "canonical_section_own"
    k: int
    metric: str = "euclidean"
    weighting: str = "unweighted"
    edge_index_convention: str = "source_to_target"
    num_nodes: int
    num_edges: int
    num_train_nodes: int
    num_val_nodes: int
    num_test_nodes: int
    num_train_train_edges: int
    num_val_val_edges: int
    num_test_test_edges: int
    num_disallowed_cross_partition_edges: int = Field(
        default=0,
        description="Forbidden edges connecting nodes in different partitions. Must be 0 in canonical protocol.",
    )
    num_disallowed_cross_section_edges: int = Field(
        default=0,
        description="Forbidden edges connecting nodes in different physical sections. Must be 0 in canonical protocol.",
    )
    edge_index_hash: str
    edge_weight_hash: str | None = None
    feature_manifest_hash: str
    created_at_utc: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
    )

    def compute_manifest_hash(self) -> str:
        return hash_dict(self.model_dump(mode="json"))


class GraphBundle:
    """Immutable graph container holding PyG connectivity tensors, partition masks, and manifests."""

    def __init__(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        train_mask: torch.Tensor,
        val_mask: torch.Tensor,
        test_mask: torch.Tensor,
        node_cell_ids: list[str],
        manifest: GraphManifest,
        edge_weight: torch.Tensor | None = None,
        node_section_ids: list[str] | None = None,
    ) -> None:
        self.edge_index = edge_index.to(dtype=torch.long)
        self.num_nodes = int(num_nodes)
        self.train_mask = train_mask.to(dtype=torch.bool)
        self.val_mask = val_mask.to(dtype=torch.bool)
        self.test_mask = test_mask.to(dtype=torch.bool)
        self.node_cell_ids = list(node_cell_ids)
        self.manifest = manifest
        self.edge_weight = edge_weight.to(dtype=torch.float32) if edge_weight is not None else None
        self.node_section_ids = (
            list(node_section_ids) if node_section_ids else ["default"] * self.num_nodes
        )

        self._validate()

    def _validate(self) -> None:
        if self.edge_index.dim() != 2 or self.edge_index.size(0) != 2:
            raise ValueError(f"edge_index must have shape [2, E], got {self.edge_index.shape}")
        if len(self.node_cell_ids) != self.num_nodes:
            raise ValueError(
                f"node_cell_ids count ({len(self.node_cell_ids)}) != num_nodes ({self.num_nodes})"
            )
        if (
            self.train_mask.size(0) != self.num_nodes
            or self.val_mask.size(0) != self.num_nodes
            or self.test_mask.size(0) != self.num_nodes
        ):
            raise ValueError("Mask sizes do not match num_nodes")
        if self.edge_weight is not None and self.edge_weight.size(0) != self.edge_index.size(1):
            raise ValueError("edge_weight length != num edges")

    def to_pyg_data(
        self,
        x: torch.Tensor | np.ndarray,
        y_train_only: torch.Tensor | np.ndarray | None = None,
    ) -> Data:
        """Convert to PyG Data container enforcing strict label isolation."""
        x_tensor = torch.as_tensor(x, dtype=torch.float32)

        data = Data(
            x=x_tensor,
            edge_index=self.edge_index,
            train_mask=self.train_mask,
            val_mask=self.val_mask,
            test_mask=self.test_mask,
            num_nodes=self.num_nodes,
        )

        if self.edge_weight is not None:
            data.edge_weight = self.edge_weight

        if y_train_only is not None:
            y_tensor = torch.as_tensor(y_train_only, dtype=torch.long)
            if y_tensor.size(0) == self.train_mask.sum().item():
                # Provided only for training nodes: pad rest with -1
                full_y = torch.full((self.num_nodes,), -1, dtype=torch.long)
                full_y[self.train_mask] = y_tensor
                data.y = full_y
            elif y_tensor.size(0) == self.num_nodes:
                masked_y = y_tensor.clone()
                masked_y[~self.train_mask] = -1
                data.y = masked_y

        return data

    def save(self, out_dir: Path | str) -> None:
        d = Path(out_dir)
        d.mkdir(parents=True, exist_ok=True)

        torch.save(self.edge_index, d / "edge_index.pt")
        if self.edge_weight is not None:
            torch.save(self.edge_weight, d / "edge_weight.pt")

        torch.save(self.train_mask, d / "train_mask.pt")
        torch.save(self.val_mask, d / "val_mask.pt")
        torch.save(self.test_mask, d / "test_mask.pt")

        (d / "node_metadata.json").write_text(
            json.dumps(
                {
                    "cell_ids": self.node_cell_ids,
                    "section_ids": self.node_section_ids,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        (d / "graph_manifest.json").write_text(
            self.manifest.model_dump_json(indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, in_dir: Path | str) -> GraphBundle:
        d = Path(in_dir)
        edge_index = torch.load(d / "edge_index.pt", map_location="cpu", weights_only=True)
        edge_weight = (
            torch.load(d / "edge_weight.pt", map_location="cpu", weights_only=True)
            if (d / "edge_weight.pt").is_file()
            else None
        )
        train_mask = torch.load(d / "train_mask.pt", map_location="cpu", weights_only=True)
        val_mask = torch.load(d / "val_mask.pt", map_location="cpu", weights_only=True)
        test_mask = torch.load(d / "test_mask.pt", map_location="cpu", weights_only=True)

        meta = json.loads((d / "node_metadata.json").read_text(encoding="utf-8"))
        manifest = GraphManifest.model_validate_json(
            (d / "graph_manifest.json").read_text(encoding="utf-8")
        )

        return cls(
            edge_index=edge_index,
            num_nodes=train_mask.size(0),
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            node_cell_ids=meta["cell_ids"],
            manifest=manifest,
            edge_weight=edge_weight,
            node_section_ids=meta.get("section_ids"),
        )
