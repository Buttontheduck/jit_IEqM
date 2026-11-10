# Final checkpoints saved on 2026-11-10 (UTC)

15 completed-run checkpoints, each at update 1,000,000:

- Experiment 19: humanoidmaze-medium, seeds 1–5 (5 checkpoints).
- Experiment 20: humanoidmaze-medium, expectiles 0.7 and 0.8, seeds 1–5 each (10 checkpoints).

Each run directory contains the original `checkpoint_1000000.pt` and `config.yaml`.
The original source files were preserved. `manifest.json` records source paths,
save timestamps, sizes and SHA-256 hashes. From this directory, verify with:

```sh
sha256sum -c SHA256SUMS
```

These are the original IEQM training checkpoints, saved by `Aware/train_ieqm.py`
on the `ieqm` branch of https://github.com/Buttontheduck/OfflineCORL (reference
commit `63ec67a`). They contain `model`, `value`, `critic`, and `target` state
dictionaries; use that training implementation and the included run configs to
restore them. They are not checkpoints produced by the newer `Aware/main.py`.

The expected checkpoint directories for the completed Experiment 21–23 sweeps
were absent when this archive was prepared. The two later benchmark reruns in
`/workspace/ieqm-full-20261110-204733` were stopped by the user before completion
and only saved through update 900,000, so they are excluded. Short parity-test
checkpoints are also excluded.
