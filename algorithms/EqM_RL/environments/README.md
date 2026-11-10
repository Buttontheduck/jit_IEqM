# Recreating the server environments

These snapshots were taken from the installed environments on 2026-11-10.
They are separate because their Python versions and package sets differ.

| Environment | Python | NumPy | Simulator packages | Purpose |
| --- | --- | --- | --- | --- |
| `hawk` | 3.10.22 | 1.23.5 | mujoco-py 2.1.2.14; mujoco 2.3.7 | D4RL training |
| `ogbench` | 3.10.0 | 1.26.4 | mujoco 3.17.0; ogbench 1.2.2 | OGBench training |
| `eagle` | 3.10.0 | 1.26.4 | mujoco 3.2.3 | Existing Eagle/Minari environment |
| `base` | 3.14.9 | — | — | Conda tooling; not used for training |

All three training environments contain the `torch==2.12.1` distribution and
CUDA 13 packages. The tested hawk/ogbench runtime reports
`torch.__version__ == '2.12.1+cu130'`, `torch.version.cuda == '13.0'`.
The server uses Ubuntu 24.04.5, Linux x86_64, an RTX 4090, and NVIDIA driver
595.91.07. The driver is external to Conda/pip.

Each directory contains:

- `requirements.txt`: every **pip-managed** package, pinned. D4RL and MJRL are
  pinned to their actual Git commits, not a moving branch or a same-named PyPI release.
- `conda-linux-64.lock.txt`: exact Conda package URLs and SHA-256 hashes,
  including Python, pip, wheel, and native libraries.
- `environment.yml`: the same Conda package versions/builds in YAML format,
  without a machine-specific prefix. It intentionally contains no pip section;
  install `requirements.txt` afterward.
- `installed-packages.json`: the full installed Python distribution inventory,
  including packages managed by Conda, for checking a restored environment.

`base/requirements.txt` contains only `docutils`: the other base packages are
managed by Conda. Installing a requirements file alone does **not** recreate
Python or the Conda libraries. Use the paired lock file or YAML.

## Restore

Install Conda on a Linux x86_64 machine, then run from `algorithms/EqM_RL`:

```bash
bash environments/restore.sh hawk
bash environments/restore.sh ogbench
# Only if you need these environments too:
bash environments/restore.sh eagle
bash environments/restore.sh base
```

This creates `hawk-recreated`, `ogbench-recreated`, etc. An optional second
argument chooses another new name. The script checks for an existing environment
and refuses to overwrite it. It does not modify the source environments.

The script restores Conda packages, installs pinned build tools, installs all
pip pins with `--no-deps --no-build-isolation`, runs `pip check`, and verifies
Python, every installed distribution version, and the Git source commits. Those flags
preserve the complete recorded package set and avoid the legacy source packages
pulling a different MJRL revision or newer build tools. Do not use them with an
incomplete requirements list. Git must be installed for hawk's two VCS packages.

Alternatively, replace the Conda-create command in the script with
`conda env create -n hawk-recreated -f environments/hawk/environment.yml` and
perform the same bootstrap and pip steps. The explicit lock avoids a new solve
and is the preferred route for this Linux platform.

Some Conda package records were subsequently overridden by pip (including
`packaging` and `setuptools` in ogbench). Both layers are preserved: restore
Conda first and pip second to recover the actual installed versions.

## D4RL's external MuJoCo installation

The `mujoco` Python wheel and the legacy `mujoco-py` engine are separate.
Hawk also needs **MuJoCo 2.1.0** installed at `~/.mujoco/mujoco210`. Download
the Linux archive and extract it there as described in the
[mujoco-py installation instructions](https://github.com/openai/mujoco-py#install-mujoco).
Our installed engine libraries' SHA-256 values are in `snapshot.json`.

The Ubuntu packages present on this server are recorded in
`system-packages.txt`. For a fresh Ubuntu host, install the build and graphics
dependencies before importing mujoco-py:

```bash
sudo apt-get update
sudo apt-get install -y git build-essential patchelf libosmesa6-dev libglew-dev libgl1 libglfw3
```

Configure the loader in each new training shell (preserve your host's NVIDIA
library paths):

```bash
export LD_LIBRARY_PATH="$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia:${LD_LIBRARY_PATH:-}"
export MUJOCO_GL=egl
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TORCHINDUCTOR_COMPILE_THREADS=2
```

The MuJoCo path is needed by hawk; OGBench uses its pinned modern MuJoCo wheel.
First import of mujoco-py may compile a native extension. Cython 0.29.37 and
setuptools 65.7.0 are deliberately preserved in hawk.

## Check the restored environments

```bash
conda run -n hawk-recreated python -m pip check
conda run -n ogbench-recreated python -m pip check
conda run -n hawk-recreated python environments/verify.py hawk
conda run -n ogbench-recreated python environments/verify.py ogbench
conda run -n hawk-recreated python -c 'import torch, d4rl, gym; e=gym.make("halfcheetah-medium-v2"); e.reset(); e.step(e.action_space.sample()); e.close(); print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
conda run -n ogbench-recreated python -c 'import torch, ogbench; e=ogbench.make_env_and_datasets("antmaze-large-navigate-singletask-v0", env_only=True); e.reset(seed=1); e.step(e.action_space.sample()); e.close(); print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
```

The original four environments passed `pip check` at snapshot time. Package
pins and VCS revisions were checked against installed metadata. This snapshot
has not yet been installed end-to-end into a clean machine; package availability,
native builds, and the host GPU driver still matter. It does not promise
bit-for-bit results across different hardware or software builds.

Datasets are also external. Copy the original D4RL cache (`~/.d4rl/datasets`)
and OGBench data (`/workspace/ogbench_data`), or configure the corresponding
dataset locations on the new host. `snapshot.json` records the hashes of the
HalfCheetah and AntMaze files used in our comparison. Log in to W&B separately,
or set `WANDB_MODE=offline`; credentials are not part of these snapshots.
