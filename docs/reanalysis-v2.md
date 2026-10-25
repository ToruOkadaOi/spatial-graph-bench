# Prediction reanalysis v2 — 2026-10-25

This is a retrospective correction to the analysis of existing predictions. It does not retrain models, change their selected epochs, or establish that the historical GNN fits satisfy the strict-inductive protocol.

## Metric policy

For each frozen split, use the intersection of labels actually present in training and test. Exclude true test cells from unseen classes uniformly across all models. Compute macro-F1 over the intersection, retaining predictions outside it as errors. Class coverage is intersection classes divided by classes present in test; cell coverage is reported separately as evaluated/test counts. Validation scores remain historical and are explicitly marked as such. No attempt is made to reselect checkpoints from test data.

For each platform, recompute the ten MLP test scores on this same population. The descriptive parity halfwidth is twice their sample SD; the hypothesis margin is `max(0.005, 2 * sample SD)`. Baseline and GNN scores pair by seed (42–51). All expected runs are required; missing runs cannot silently shrink the family.

## Inference amendment

The v2 family consists of 112 comparisons (four platforms × four GNNs × seven topologies), each with superiority, inferiority and equivalence hypotheses: 336 tests. Superiority/inferiority test beyond ±epsilon; equivalence uses TOST. Holm adjusted p-values use the cumulative maximum of sorted, multiplicity-scaled p-values, enforcing the step-down stopping rule. No minimum-p hypothesis selection is performed. This conservative family includes secondary bipartite/shuffled comparisons; their protocol identities remain separate in the output. This is a new, explicitly retrospective rule, not a claim of original preregistration.

Outcomes are Positive lift, Negative lift, Equivalent, or Inconclusive. A nonsignificant test does not establish equivalence. Reported 90% paired t intervals are unadjusted descriptive intervals, not simultaneous confidence intervals. The uncertainty concerns optimization seeds on one fixed biological holdout; it does not estimate between-donor or between-tissue variability. Per-class and convex-hull boundary diagnostics are descriptive, without inferential per-class parity claims.

## Provenance and historical preservation

Original run metrics, predictions, probabilities, training histories, feature manifests and graph manifests are retained unchanged. Corrected outputs go to `results/reports/reanalysis_v2/`; narrative, CSVs and synthesis tables are generated from them. Historical versions of edited documentation remain in Git.

`audits/artifact_integrity_v2.json` anchors the bytes of 6,056 original files and explicitly lists old PCA hash discrepancies. Several v1 manifests hashed PCA values before their float32 serialization. The v2 index records saved file bytes without rewriting the old identity chain; it does not assert that current files prove every detail of the historical runtime. This byte index is trusted through repository review/version control, not through its self-checksum alone. Verification never automatically regenerates or reseals it. New preprocessing hashes the persisted float32 arrays.

The pipeline verifies the index, cell order, split identity, label mapping, graph contents and run-parent identities before analysis. Original result files are read-only inputs. A failed integrity check is an error, not permission to overwrite or migrate evidence.

## Remaining training work

Historical GNN training pooled train/validation/test features through BatchNorm and selected epochs using stale training-mode validation outputs. Shuffled graph controls mix coordinates across sections; rewired controls lose the original undirected reciprocity. Reanalysis cannot repair these learned models or recover unsaved historical weights. Corrected strict-inductive claims require new GNN fits and corrected controls. The current training algorithms have not been changed by this analysis correction.

Future-run safeguards now preserve resolved configurations and selected neural weights, reject incompatible caches, and refuse overwriting an existing run directory. These safeguards do not retrofit missing historical source diffs, weights, fitted PCA/scaler objects, or a tuning study. Baselines are described as fixed configurations rather than as supported by an unrecorded tuning budget.

## Commands

```bash
./scripts/reproduce_all.sh                     # fetch missing assets; verify; analyze; plot
./scripts/reproduce_all.sh --offline           # use existing files or local release archives
./scripts/reproduce_all.sh --offline --no-figures
uv run --frozen pytest tests
```

The reproduction command does not train or publish. MERFISH, Stereo-seq and Open-ST legacy GPU archives omit baselines (Xenium includes them). Four local Stereo-seq runs also differ from the archived versions. The checked-in `audits/reanalysis_recovery_v2.zip` supplies exact local historical manifests, metrics and test predictions for these gaps, checked against the byte index. Its JSON manifest records each reason; neither local original runs nor old archives are changed. Probabilities and training histories are optional for analysis; if present, their byte anchors are also verified.

Release archives are verified against digests in `configs/release_assets.json`; extracted originals are also checked against the v2 index. Online recovery requires GitHub CLI access to the repository's releases. Downloads and environment installation affect runtime. CI and Docker use `uv.lock`; exact historical GPU bitwise reproducibility is still unverified.
