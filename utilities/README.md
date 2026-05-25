# tools

Off-pipeline utilities. Nothing here is required to reproduce the
trained model; everything here helps with diagnostics or post-training
experimentation.

## Utilities

- `profile_training.py` — training-loop bottleneck profiler. Wraps a
  short training run with PyTorch profiler hooks and prints per-step
  time spent in forward, backward, optimizer, and data loading. Use
  this to diagnose throughput regressions on a new GPU or after a
  framework upgrade.
- `slerp_merge.py` — multi-checkpoint SLERP merger. Given a list of
  checkpoint paths and matching weights, produces a single merged
  checkpoint by spherical linear interpolation across the parameter
  tensors. Used by the SLERP ablation; see
  `docs/REPRODUCING_ABLATIONS.md` for the two regimes (merging across
  data-composition runs, merging across consecutive checkpoints of a
  single run).

Both scripts accept `--help`.
