"""Stage edited Toolathlon source files for the agent phase and restore them for scoring."""

from __future__ import annotations

import argparse
import json
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path

HARNESS_PATHS = (
    "scripts/containerized/container_agent.py",
    "utils/__init__.py",
    "utils/status_manager.py",
    "utils/api_model",
    "utils/aux_tools",
    "utils/conversation",
    "utils/data_processing",
    "utils/data_structures",
    "utils/general",
    "utils/local_servers",
    "utils/logging",
    "utils/mcp",
    "utils/misc",
    "utils/openai_agents_monkey_patch/custom_mcp_util.py",
    "utils/openai_agents_monkey_patch/custom_run_impl.py",
    "utils/openai_agents_monkey_patch/tool_name_aliases.py",
    "utils/roles",
    "utils/system_prompts",
    "utils/task_runner",
    "configs/mcp_servers",
)


def harness_files(root: Path) -> dict[str, Path]:
    files = {}
    for scope in HARNESS_PATHS:
        target = root / scope
        paths = target.rglob("*") if target.is_dir() else (target,)
        for path in paths:
            relative = path.relative_to(root)
            if any(part.startswith(".") or part == "__pycache__" for part in relative.parts):
                continue
            if path.is_symlink() or any(parent.is_symlink() for parent in path.parents if parent.is_relative_to(root)):
                raise ValueError(f"Harness path must not traverse symlinks: {relative}")
            if path.is_file():
                files[relative.as_posix()] = path
    return files


def changed_files(candidate: Path, baseline: Path) -> tuple[list[str], list[str]]:
    before, after = harness_files(baseline), harness_files(candidate)
    paths = sorted(
        name
        for name in before.keys() | after.keys()
        if name not in before
        or name not in after
        or before[name].read_bytes() != after[name].read_bytes()
        or stat.S_IMODE(before[name].stat().st_mode) != stat.S_IMODE(after[name].stat().st_mode)
    )
    return paths, [name for name in paths if name in after]


def command(*args: str) -> str:
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT, timeout=60)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Agent file staging failed ({args[1]}): {exc.output[-3000:]}") from exc


# These run in the task container using its original interpreter, before imports
# of candidate modules. The backup and manifest stay outside that container.
_INSPECT = """
import json, pathlib, sys
result = {}
for name in json.loads(sys.argv[1]):
    path = pathlib.Path('/workspace') / name
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('Harness path must not traverse symlinks: ' + name)
    if path.exists() and not path.is_file():
        raise ValueError('Harness target must be a regular file: ' + name)
    result[name] = path.exists()
print(json.dumps(result))
"""
_REMOVE = """
import importlib.util, json, pathlib, shutil, sys
for name in json.loads(sys.argv[1]):
    path = pathlib.Path('/workspace') / name
    if any(p.is_symlink() for p in path.parents):
        raise ValueError('Cannot restore harness through a symlink: ' + name)
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)
    if path.suffix == '.py':
        pathlib.Path(importlib.util.cache_from_source(str(path))).unlink(missing_ok=True)
    path.parent.mkdir(parents=True, exist_ok=True)
"""


def stage(runtime: str, container: str, candidate: Path, baseline: Path, backup: Path) -> None:
    paths, replacements = changed_files(candidate, baseline)
    backup.mkdir(parents=True, exist_ok=False)
    manifest = (
        json.loads(
            command(runtime, "exec", container, "/workspace/.venv/bin/python", "-c", _INSPECT, json.dumps(paths))
        )
        if paths
        else {}
    )
    for name, existed in manifest.items():
        if existed:
            target = backup / "files" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            command(runtime, "cp", f"{container}:/workspace/{name}", str(target))
    (backup / "manifest.json").write_text(json.dumps(manifest))
    if not paths:
        return
    command(runtime, "exec", container, "/workspace/.venv/bin/python", "-c", _REMOVE, json.dumps(paths))
    if replacements:
        with tempfile.TemporaryDirectory(dir=backup, prefix="candidate-") as staging:
            root = Path(staging)
            for name in replacements:
                destination = root / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(candidate / name, destination)
            command(runtime, "cp", f"{root}/.", f"{container}:/workspace")


def restore(runtime: str, container: str, backup: Path) -> None:
    manifest = json.loads((backup / "manifest.json").read_text())
    if not manifest:
        return
    command(runtime, "exec", container, "/workspace/.venv/bin/python", "-c", _REMOVE, json.dumps(list(manifest)))
    for name, existed in manifest.items():
        if existed:
            command(runtime, "cp", str(backup / "files" / name), f"{container}:/workspace/{name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("stage", "restore"))
    parser.add_argument("runtime")
    parser.add_argument("container")
    parser.add_argument("backup", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    if args.phase == "stage":
        if args.candidate is None or args.baseline is None:
            parser.error("stage requires --candidate and --baseline")
        stage(args.runtime, args.container, args.candidate, args.baseline, args.backup)
    else:
        restore(args.runtime, args.container, args.backup)
