"""Check a restored environment against its installed Python package snapshot."""
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import re
import sys


def canonical(name):
    return re.sub(r"[-_.]+", "-", name).lower()


if len(sys.argv) != 2 or sys.argv[1] not in {"base", "hawk", "eagle", "ogbench"}:
    sys.exit("Usage: python verify.py base|hawk|eagle|ogbench")

directory = Path(__file__).resolve().parent / sys.argv[1]
snapshot = json.loads((directory / "installed-packages.json").read_text())
expected = {canonical(k): v for k, v in snapshot["packages"].items()}
actual = {canonical(d.metadata["Name"]): d.version for d in metadata.distributions()}
errors = []
if platform.python_version() != snapshot["python"]:
    errors.append(f"Python: expected {snapshot['python']}, got {platform.python_version()}")
for name in sorted(expected.keys() | actual.keys()):
    if expected.get(name) != actual.get(name):
        errors.append(f"{name}: expected {expected.get(name)}, got {actual.get(name)}")

for line in (directory / "requirements.txt").read_text().splitlines():
    match = re.fullmatch(r"(\S+) @ git\+(.+)@([0-9a-f]{40})", line)
    if match:
        name, url, commit = match.groups()
        try:
            source = json.loads(metadata.distribution(name).read_text("direct_url.json") or "{}")
        except metadata.PackageNotFoundError:
            source = {}
        if source.get("url") != url or source.get("vcs_info", {}).get("commit_id") != commit:
            errors.append(f"{name}: Git source or commit differs from the snapshot")

if errors:
    sys.exit("\n".join(errors))
print(f"{sys.argv[1]}: Python {snapshot['python']}, all {len(expected)} package versions and Git pins match.")
