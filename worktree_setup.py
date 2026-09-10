"""Symlink the main checkout's suite venvs and caches into a worktree.

Symlinks ``GPU_ODE_*/venv``, ``GPU_ODE_MPGOS/build_cache`` and
``generated``; copies ``.claude/settings.local.json``; aborts on a
junction. Env: ``ORCA_WORKTREE_PATH``, ``ORCA_ROOT_PATH``.
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
    """Directory symlink; on Windows this needs Developer Mode."""
    link.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(target, link, target_is_directory=True)
    except OSError as error:
        if os.name == "nt":
            raise SystemExit(
                f"cannot symlink {link}: {error}; enable Windows Developer "
                "Mode (Settings > System > For developers) and rerun")
        raise


def share_directories(root, worktree):
    for relative in SHARED:
        source = root / relative
        link = worktree / relative
        if not source.is_dir():
            print(f"absent     {relative} (not built in {root})")
            continue
        target = source.resolve()
        if os.path.isjunction(link):
            raise SystemExit(
                f"{link} is a junction; replace it with a symlink before "
                f"continuing, or git worktree remove will delete {target}")
        if link.is_symlink() or link.exists():
            print(f"exists     {relative}")
            continue
        link_directory(target, link)
        print(f"symlink    {relative} -> {target}")


def copy_local_files(root, worktree):
    """Copy the gitignored Claude settings into the worktree."""
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
