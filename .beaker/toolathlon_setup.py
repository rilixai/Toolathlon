"""Prepare the pinned Toolathlon benchmark in local and hosted runs."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from toolathlon_diagnostics import phase

REPOSITORY = Path(__file__).resolve().parents[1]
UPSTREAM_REVISION = "9be8d8fe07a497b18ee61e3f2ae694e9797f39eb"
# This commit contains the migrated agent, trusted runner, and original graders.
BASELINE_COMMIT = "09b8cd5c2d3bb9d759b7e2883d0981bb1ec0f56e"
BASELINE_REPOSITORY = "https://github.com/rilixai/Toolathlon.git"


# MCP servers and local tools backed by third-party accounts (tokens in upstream
# configs/token_key_session.py). Canvas, emails, WooCommerce and k8s are local containers.
EXTERNAL_ACCOUNT_SERVERS = frozenset(
    {
        "github",
        "google-cloud",
        "google_calendar",
        "google_forms",
        "google_map",
        "google_sheet",
        "huggingface",
        "notion",
        "snowflake",
        "wandb",
        "youtube",
        "youtube-transcript",
    }
)
EXTERNAL_ACCOUNT_LOCAL_TOOLS = frozenset({"web_search"})


def benchmark_cache() -> Path:
    root = Path(os.environ.get("BEAKER_TOOLATHLON_CACHE", "/tmp"))
    return (root / f"beaker-toolathlon-fork-{BASELINE_COMMIT[:12]}").resolve()


def runtime_env(source: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.pop("VIRTUAL_ENV", None)
    env["PATH"] = os.pathsep.join((str(Path(sys.executable).parent), env.get("PATH", os.defpath)))
    env["UV_PROJECT_ENVIRONMENT"] = str(source / ".venv")
    env["UV_FROZEN"] = "1"
    return env


def prepare_toolathlon() -> tuple[Path, Path]:
    """Keep the trusted benchmark independent of every candidate checkout."""
    cache = benchmark_cache()
    source = cache / "upstream"
    cache.mkdir(parents=True, exist_ok=True)
    with (cache / "setup.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not source.exists():
            staging = cache / "upstream.incomplete"
            shutil.rmtree(staging, ignore_errors=True)
            staging.mkdir()
            try:
                with phase("fetch pinned Toolathlon baseline"):
                    subprocess.run(["git", "init", "--quiet", str(staging)], check=True)
                    subprocess.run(
                        ["git", "-C", str(staging), "fetch", "--depth=1", BASELINE_REPOSITORY, BASELINE_COMMIT],
                        check=True, timeout=900,
                    )
                    subprocess.run(
                        ["git", "-C", str(staging), "checkout", "--quiet", "--detach", BASELINE_COMMIT], check=True
                    )
                staging.rename(source)
            finally:
                if staging.exists():
                    shutil.rmtree(staging)
        commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        if commit != BASELINE_COMMIT:
            raise RuntimeError(f"Expected trusted Toolathlon baseline {BASELINE_COMMIT}, found {commit}")
        ready = cache / "host-dependencies-ready"
        if not ready.is_file() or not (source / ".venv/bin/python").is_file():
            with phase("prepare Toolathlon host dependencies"):
                env = runtime_env(source)
                env.pop("UV_FROZEN")
                subprocess.run(["uv", "sync", "--locked"], cwd=source, env=env, check=True, timeout=600)
            ready.touch()
    return cache, source


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def groundtruth_hashes(source: Path, task: str) -> dict[str, str]:
    task_dir = source / "tasks/finalpool" / task
    gt_dir = task_dir / "groundtruth_workspace"
    if gt_dir.is_dir():
        hashes = {
            path.relative_to(gt_dir).as_posix(): _sha256(path)
            for path in sorted(gt_dir.rglob("*"))
            if path.is_file() and "__pycache__" not in path.parts
        }
        if hashes:
            return hashes
    eval_dir = task_dir / "evaluation"
    if not eval_dir.is_dir():
        raise ValueError(f"Missing evaluation directory for {task}: {eval_dir}")
    hashes = {
        f"evaluation/{path.relative_to(eval_dir).as_posix()}": _sha256(path)
        for path in sorted(eval_dir.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts
    }
    if not hashes:
        raise ValueError(f"No groundtruth or evaluation files found in {task_dir}")
    return hashes


def load_task_reference(source: Path, task: str) -> dict[str, Any]:
    """Read the official task inputs and scoring references for a platform dataset row."""
    if not re.fullmatch(r"[A-Za-z0-9_-]+", task):
        raise ValueError(f"Unsupported Toolathlon task: {task}")
    task_dir = source / "tasks/finalpool" / task
    instruction_path = task_dir / "docs/task.md"
    config_path = task_dir / "task_config.json"
    eval_path = task_dir / "evaluation/main.py"
    if not instruction_path.is_file() or not config_path.is_file() or not eval_path.is_file():
        raise ValueError(f"Task {task} is missing required files in {task_dir}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    instruction = instruction_path.read_text(encoding="utf-8").strip()
    gt_hashes = groundtruth_hashes(source, task)
    return {
        "id": task,
        "input": {
            "task": task,
            "instruction": instruction,
            "mcp_servers": list(config.get("needed_mcp_servers", [])),
            "local_tools": list(config.get("needed_local_tools", [])),
        },
        "expected": {
            "task": task,
            "upstream_revision": UPSTREAM_REVISION,
            "evaluation_module": f"tasks.finalpool.{task}.evaluation.main",
            "groundtruth_hashes": gt_hashes,
        },
    }
