"""Prepare a fresh git worktree of GPUODEBenchmarks for agent work.

Orca runs this from the new worktree after ``worktree create`` (see
``orca.yaml``); it also runs by hand from any worktree::

    python worktree_setup.py

The suite environments are built once per machine by
``setup_all_environments.py`` and install nothing editable from this
repo, so every worktree links to the main checkout's copies instead of
rebuilding them: the ``GPU_ODE_*/venv`` directories, the MPGOS build
cache, and the cubie codegen cache under ``generated``. Julia uses the
tracked ``Project.toml`` against the shared depot and needs nothing.
Run logs stay per worktree.

Environment:

``ORCA_WORKTREE_PATH``
    The worktree to prepare (default: this file's directory).
``ORCA_ROOT_PATH``
    The main checkout (default: the owner of the shared ``.git``).
"""

import os
import shutil
import subprocess
from pathlib import Path

SHARED = (
    Path("GPU_ODE_CUBIE") / "venv",
    Path("GPU_ODE_CUBIE_MLIR") / "venv",
    Path("GPU_ODE_JAX") / "venv",
    Path("GPU_ODE_PyTorch") / "venv",
    Path("GPU_ODE_MYOKIT_CUDA") / "venv",
    Path("GPU_ODE_MPGOS") / "build_cache",
    Path("generated"),
)
LOCAL_FILES = (Path(".claude") / "settings.local.json",)


def worktree_path():
    override = os.environ.get("ORCA_WORKTREE_PATH")
    if override:
        return Path(override).resolve()
    return Path(__file__).resolve().parent


def root_path(worktree):
    override = os.environ.get("ORCA_ROOT_PATH")
    if override:
        return Path(override).resolve()
    common = subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        cwd=worktree, check=True, capture_output=True, text=True,
    ).stdout.strip()
    return Path(common).resolve().parent


def link_directory(target, link):
    """Directory symlink, or a junction where Windows refuses symlinks."""
    link.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(target, link, target_is_directory=True)
        return "symlink"
    except OSError:
        if os.name != "nt":
            raise
    subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)],
                   check=True, capture_output=True)
    return "junction"


def share_directories(root, worktree):
    for relative in SHARED:
        source = root / relative
        link = worktree / relative
        if not source.is_dir():
            print(f"absent     {relative} (not built in {root})")
            continue
        target = source.resolve()
        if link.is_symlink() or link.exists():
            print(f"exists     {relative}")
            continue
        kind = link_directory(target, link)
        print(f"{kind:<10} {relative} -> {target}")


def copy_local_files(root, worktree):
    """Gitignored per-repo Claude settings travel with the worktree."""
    for relative in LOCAL_FILES:
        source = root / relative
        target = worktree / relative
        if source.is_file() and not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            print(f"copied {source} -> {target}")


def verify(worktree):
    cubie_python = worktree / "GPU_ODE_CUBIE" / "venv"
    cubie_python /= "Scripts/python.exe" if os.name == "nt" else "bin/python"
    if not cubie_python.is_file():
        raise SystemExit(f"{cubie_python} is missing; run "
                         "setup_all_environments.py in the main checkout")
    result = subprocess.run(
        [str(cubie_python), "-c", "import cubie; print(cubie.__version__)"],
        check=True, capture_output=True, text=True, cwd=worktree)
    print(f"cubie      {result.stdout.strip()} via {cubie_python}")


def main():
    worktree = worktree_path()
    root = root_path(worktree)
    print(f"worktree   {worktree}")
    print(f"root       {root}")
    share_directories(root, worktree)
    copy_local_files(root, worktree)
    verify(worktree)


if __name__ == "__main__":
    main()
