"""Distribution, workspace and import versions must describe the same release."""

import ast
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


def test_python_rust_and_import_versions_are_consistent():
    root = Path(__file__).resolve().parents[1]
    with (root / "pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle)
    with (root / "Cargo.toml").open("rb") as handle:
        cargo = tomllib.load(handle)
    module = ast.parse((root / "src/ov_radiomics/__init__.py").read_text())
    versions = [
        ast.literal_eval(node.value)
        for node in module.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "__version__"
            for target in node.targets
        )
    ]
    assert len(versions) == 1
    assert project["project"]["version"] == cargo["workspace"]["package"]["version"]
    assert versions[0] == project["project"]["version"]
