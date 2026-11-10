#!/usr/bin/env bash
# Restore into a new Conda environment. Existing training environments are untouched.
set -euo pipefail

source_env=${1:?Usage: bash restore.sh hawk|ogbench|eagle|base [new-environment-name]}
target_env=${2:-${source_env}-recreated}
case "$source_env" in
  hawk|ogbench|eagle|base) ;;
  *) echo "Unknown snapshot: $source_env" >&2; exit 2 ;;
esac
snapshot_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
conda_command=${CONDA_EXE:-conda}
snapshot_python=${CONDA_PYTHON_EXE:-python}
case "$target_env" in
  base|*/*) echo "Choose a new environment name, not base or a path." >&2; exit 2 ;;
esac
"$conda_command" env list --json | "$snapshot_python" -c '
import json, pathlib, sys
target = sys.argv[1]
if any(pathlib.Path(p).name == target for p in json.load(sys.stdin)["envs"]):
    sys.exit(f"Environment already exists: {target}; choose a new name.")
' "$target_env"

"$conda_command" create --yes --name "$target_env" \
  --file "$snapshot_dir/$source_env/conda-linux-64.lock.txt"

# Bootstrap source-build dependencies before installing legacy Gym/D4RL/MJRL.
# wheel and pip themselves were restored from Conda above.
case "$source_env" in
  hawk)
    "$conda_command" run --no-capture-output -n "$target_env" python -m pip install \
      --no-deps --no-build-isolation setuptools==65.7.0 Cython==0.29.37 numpy==1.23.5
    ;;
  eagle|ogbench)
    "$conda_command" run --no-capture-output -n "$target_env" python -m pip install \
      --no-deps --no-build-isolation setuptools==81.0.0 numpy==1.26.4
    ;;
esac

# Every pip dependency is already pinned. Avoid following the legacy projects'
# transitive VCS references to moving branches or upgrading build dependencies.
"$conda_command" run --no-capture-output -n "$target_env" python -m pip install \
  --no-deps --no-build-isolation -r "$snapshot_dir/$source_env/requirements.txt"
"$conda_command" run --no-capture-output -n "$target_env" python -m pip check
"$conda_command" run --no-capture-output -n "$target_env" python "$snapshot_dir/verify.py" "$source_env"
echo "Restored $source_env as $target_env. See README.md for system libraries and smoke checks."
