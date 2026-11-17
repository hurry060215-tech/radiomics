"""Enumerate public source files without inspecting ignored developer artifacts."""

import os
import subprocess
from pathlib import Path

BUILD_DIRECTORIES = {
    ".git",
    ".venv",
    ".docs-env",
    ".build-tools",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "__pycache__",
    "_build",
    "target",
    "build",
    "dist",
    "artifacts",
    "package-assets",
    ".ipynb_checkpoints",
}


def public_files(root):
    root = Path(root).resolve()
    # A .git file is also valid (linked worktrees). Never borrow a parent repo.
    if (root / ".git").exists():
        result = subprocess.run(
            ["git", "ls-files", "-z"], cwd=root, capture_output=True, check=True
        )
        paths = [root / os.fsdecode(name) for name in result.stdout.split(b"\0") if name]
        return sorted(path for path in paths if path.is_file())
    # Prepared sdists and source archives carry no index. Prune build products
    # before descending so this also works after a local build of an archive.
    paths = []
    for directory, names, files in os.walk(root, followlinks=False):
        names[:] = [name for name in names if name not in BUILD_DIRECTORIES]
        for name in files:
            path = Path(directory) / name
            if not path.is_symlink() and path.resolve().is_relative_to(root):
                paths.append(path)
    return sorted(paths)
