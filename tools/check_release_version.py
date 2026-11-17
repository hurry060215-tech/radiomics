"""Validate release metadata without importing or building the native package."""

import argparse
import ast
import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


def versions(root):
    root = Path(root)
    python = ast.parse((root / "src/ov_radiomics/__init__.py").read_text(encoding="utf-8"))
    python_version = next(
        ast.literal_eval(node.value)
        for node in python.body
        if isinstance(node, ast.Assign)
        and any(isinstance(name, ast.Name) and name.id == "__version__" for name in node.targets)
    )
    citation = re.search(
        r'^version:\s*[\'"]?([^\s\'"]+)',
        (root / "CITATION.cff").read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    return {
        "pyproject.toml": tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))[
            "project"
        ]["version"],
        "Cargo.toml": tomllib.loads((root / "Cargo.toml").read_text(encoding="utf-8"))["workspace"][
            "package"
        ]["version"],
        "__init__.py": python_version,
        "CITATION.cff": citation.group(1) if citation else None,
    }


def check(root, expected):
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", expected):
        raise ValueError("Use an explicit stable version, for example 0.1.0.")
    actual = versions(root)
    if any(value != expected for value in actual.values()):
        raise ValueError(f"Release version mismatch: requested {expected}; {actual}")
    return actual


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    print(check(args.root, args.version))
