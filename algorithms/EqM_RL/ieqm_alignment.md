# IEqM alignment

`Aware/` and `Aware_Minari/` retain the local Aware structure while matching the
trained EqM actor and IFQL critic in
`Aware/train_ieqm.py` from fetched commit
`63ec67a09f29b17f64d8b00efe22a42af3f6b895` (`origin/ieqm`). The defaults match
its D4RL Experiment 23, with expectile 0.9. Use `expectile=0.8` for the other variant.

## Local structure retained

Networks and the shared optimizer are constructed explicitly in `main.py`, then
passed to `Actor` and `Otter`. There is no `Otter.create()` factory. The existing
`TwinCritic`, `VCritic`, `QCriticMLP`, and `VCriticMLP` classes remain, including
their `[B, 1]` outputs. The replay buffer keeps its original column-shaped rewards
and terminal masks.

`Otter` keeps `_bc_actor_loss`, `_v_critic_loss`, and `_q_critic_loss` as separate
methods. A small `_total_loss` method combines them for the shared optimizer and
compilation. The two separate optimizer-update methods are replaced by one joint
update because the fetched IEqM algorithm uses one Adam step.

Configuration retains the nested `actor` and `agent` sections. Evaluation stays
inside `train()` as `eval_actor`, and the existing tensor-aware logger, run naming,
progress bar, and checkpoint flow remain. Loss names are still
`train/q_critic_loss`, `train/v_critic_loss`, and `train/actor_loss`.

Actor normalization lives in `utils/helper.py`: `StateNormalizer` stores the
dataset statistics and calls `normalize_state_tensor`. The training losses use
the same function. Sampling keeps raw states for the critics and passes a
normalized copy to the EqM field.

## Numerical changes retained

- Q1, Q2, and V use four 512-unit GELU/LayerNorm layers, Xavier weights, and zero
  biases. The EqM field keeps Aware's Mish network, also at four 512-unit layers.
- V and Q receive raw observations. Only the actor receives normalized observations.
  Rewards stay unchanged; dataset actions are clipped just inside the action bounds.
- Random initialization follows V, Q1/Q2, then the actor. Target Q is copied inside
  `Otter`, which consumes no random draws. One Adam optimizer updates the sum of
  V, Q, and unweighted EqM behavior-cloning losses.
- Target Q moves toward the **pre-update** Q parameters, before `optimizer.step()`.
- `compile=true` compiles the combined training loss with `reduce-overhead`.
  Evaluation uses the uncompiled NAG sampler, as in the fetched trainer.
- Separate generators sample batches (`seed + 1000`) and evaluation noise
  (`seed + 2000`). Evaluation runs episodes together and selects the highest-Q
  candidate from 32 actions per state.
- Training starts at step 1. `num_train_ops=1000001` means 1,000,000 updates.
  Evaluation runs every 100,000 updates and on the last update. The final score
  averages the last three evaluations. Training metrics sample every 10 updates
  and average at each logging interval.
- Checkpoints retain the local keys: `model`, `q_nets`, `target_q_nets`, `v_net`.
  The tests translate these names to the reference schema when comparing tensors.
  Earlier checkpoints with different network sizes cannot load into the new defaults.

Shared algorithm code matches between the two folders; comments and formatting
may differ. Data helpers,
loading, terminal/truncation handling, environment calls, dataset names, and score
normalization differ. The fetched
trainer's optional frozen-actor comparison is outside these two entry points.
`Aware/main.py` also supports OGBench single-task datasets through
`utils/ogbench_data.py`, preserving the same actor, critics, and training loop.

## Running

For the single-seed comparisons against completed W&B Experiments 23 and 21,
see [Aware/configs/README.md](Aware/configs/README.md). Those configs include
server checkpoint paths and the exact selected training and evaluation settings.

In the appropriate environment with the project's dependencies installed, run from
`Aware/` for D4RL or `Aware_Minari/` for Minari:

```bash
python main.py device=cuda compile=true checkpoints_path=./checkpoints
```

Use the local configuration names, for example `expectile=0.8`,
`actor.num_step=20`, `agent.scale_gradient=8.0`, and `num_actions_inference=32`.
`agent.expectile` defaults to `${expectile}`. For an eager CPU run, use
`device=cpu compile=false`. The original local checkpoint path is restored; override
it when running on another machine.

The reference comparison uses the shipped settings: raw rewards
(`normalize_reward=false`), plain EqM BC (`agent.fino=false`), and NAG sampling.
The original optional helpers remain available for experiments; changing those
settings can change the algorithm and invalidate the reference comparison.

## Verification and limits

From `EqM_RL/`:

```bash
python -m unittest discover -s tests -p 'test_ieqm_reference.py' -v
```

All eight tests pass. They read source directly
from the pinned Git commit and compare initialization,
losses, gradients, updates, random streams, candidate selection, logged metrics,
evaluation, and saved parameters. D4RL, Minari, and OGBench training paths run on synthetic
transitions and deterministic test environments; dataset downloads and external
logging/configuration services are replaced in those integration tests. OGBench
tests also cover task masks, per-task reset seeds, success at the final step,
and preservation of NumPy's global random state.

CPU eager comparisons require exact equality. The compile test checks full-graph
tracing with the eager backend, not Inductor code generation or CUDA graphs. CUDA
and real D4RL/Minari/OGBench benchmark runs were not verified on this machine; its CPU
Inductor attempt was blocked by the local `backports.tarfile` dependency error.

Matching code does not make different datasets or simulators produce identical
losses and returns. Comparing runs also requires the same data order, settings,
seed, device, PyTorch/compiler versions, and compilation mode. In particular,
Minari's `simple-v0` is not D4RL's `medium-v2` dataset.
