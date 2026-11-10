# Single-run IEqM comparisons

These configs match the training and evaluation settings of two completed W&B
runs, checked against their saved configurations and the fetched `origin/ieqm`
source at `63ec67a09f29b17f64d8b00efe22a42af3f6b895`.

| Config | Reference | Environment | Seed | Reference final score |
| --- | --- | --- | --- | --- |
| `config_d4rl.yaml` | [Experiment 23](https://wandb.ai/duck13/E/runs/07fbf847-9784-439b-9971-5b0b4897bd52) | `halfcheetah-medium-v2` | 1 | 55.28248762297928 |
| `config_og.yaml` | [Experiment 21](https://wandb.ai/duck13/E/runs/54980a53-1a71-4cba-9c9b-f44eb10d58e9) | `antmaze-large-navigate-singletask-v0`, task 1 | 1 | 31.333333333333332% |

Both use expectile 0.9, gamma 0.99, tau 0.005, learning rate 0.0003,
batch size 256, four 512-unit layers for actor and critics, and compiled training
with `reduce-overhead`. Sampling uses 20 NAG steps, step size 0.01, momentum 0.3,
and 32 candidate actions. EqM uses gradient scale 8 and truncation threshold 0.8.
The actor loss is unweighted behavior cloning; `temperature: null` matches the
reference metadata and is unused by this training path.

`num_train_ops: 1000001` means 1,000,000 updates. Evaluation and checkpoints are
every 100,000 updates. Each evaluation uses 50 episodes, test seed 69; the final
score averages evaluations at 800k, 900k, and 1M. Training statistics are sampled
every 10 updates and logged every 5,000. The D4RL score is normalized return;
the OGBench score is success percentage on the dataset's own task (`eval_tasks: own`).
Neither selected reference run uses a frozen BC actor.

## Server commands

From `OfflineCORL/algorithms/EqM_RL/Aware`, run each command in the corresponding
existing server environment used for the reference:

```bash
conda activate hawk
WANDB_ENTITY=duck13 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python main.py --config-path configs --config-name config_d4rl
```

```bash
conda activate ogbench
WANDB_ENTITY=duck13 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python main.py --config-path configs --config-name config_og
```

Keep the reference server's MuJoCo setup for `hawk`, including its library path.
OGBench data defaults to `/workspace/ogbench_data`; override with
`ogbench_dir=/your/dataset/path` if needed. New run names, groups, and checkpoint
folders distinguish these comparisons from the reference runs. Each launch
gets a fresh W&B run ID. Running `python main.py` alone still uses `Aware/config.yaml`.

Compare `train/q_critic_loss` with the reference's `train/q_loss`, and
`train/v_critic_loss` with `train/v_loss`. Actor loss and evaluation metric names
already match. Compare at the same logged update number.

The reference D4RL dataset has 999,000 transitions, 17 observation dimensions,
and 6 action dimensions. The OGBench dataset has 1,000,000 transitions,
29 observation dimensions, and 8 action dimensions. OGBench task terminals
come from `1 - masks`, not the trajectory-boundary `terminals` field.

The local tests compare losses, updates, evaluation, and checkpoints exactly on
CPU synthetic data, including the OGBench loader and evaluator. Actual CUDA
benchmark runs remain to be compared on the server. Matching these settings
does not guarantee bit-for-bit GPU results: use the same dataset files, GPU,
PyTorch/CUDA/compiler versions, and simulator environment as the reference.
