"""Generate apidoc/autosummary pages for the explicit public export list only.

The package is inspected as source, never imported here. Optional lazy result
aliases receive static domain pages rather than importing a reference backend.
"""

import argparse
import ast
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OPTIONAL_ALIASES = {"TrainingResult", "FusionResult"}


def public_exports(root=ROOT):
    package = Path(root) / "src/ov_radiomics"
    tree = ast.parse((package / "__init__.py").read_text(encoding="utf-8"))
    assignments = {
        target.id: node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    routes = {
        name: module
        for module, names in ast.literal_eval(assignments["_EXPORTS"]).items()
        for name in names
    }
    routes.update(
        {
            alias.asname or alias.name: node.module
            for node in tree.body
            if isinstance(node, ast.ImportFrom) and node.level == 1
            for alias in node.names
        }
    )
    names = []
    for item in assignments["__all__"].elts:
        if isinstance(item, ast.Constant) and isinstance(item.value, str):
            names.append(item.value)
        elif (
            isinstance(item, ast.Starred)
            and isinstance(item.value, ast.Name)
            and item.value.id == "_LOOKUP"
        ):
            names.extend(name for name in routes if name not in names)
        else:
            raise ValueError("Public exports must remain an explicit list of routed names.")
    normal, optional = [], []
    for name in names:
        module_tree = ast.parse((package / (routes[name] + ".py")).read_text(encoding="utf-8"))
        definitions = {
            node.name
            for node in module_tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        }
        if name in definitions:
            normal.append(name)
        elif name in OPTIONAL_ALIASES:
            optional.append(name)
        else:
            raise ValueError(
                "Document the lazy public alias without importing its backend: " + name
            )
    return normal, optional


def build(root=ROOT, output=None):
    from sphinx.ext.apidoc import main as apidoc

    root = Path(root).resolve()
    output = Path(output).resolve() if output else root / "website/reference/api"
    package = root / "src/ov_radiomics"
    normal, optional = public_exports(root)
    summary = (
        "Public Python API\n=================\n\n"
        "This reference is generated from ``ov_radiomics.__all__``. "
        "It does not discover or import optional backend implementations.\n\n"
        ".. currentmodule:: ov_radiomics\n\n"
        ".. autosummary::\n   :toctree: generated\n   :nosignatures:\n\n"
        + "".join("   " + name + "\n" for name in normal)
        + "\nOptional Python result aliases\n------------------------------\n\n"
        ".. toctree::\n   :maxdepth: 1\n\n"
        + "".join("   ov_radiomics." + name + "\n" for name in optional)
    )
    with tempfile.TemporaryDirectory(prefix="radiomics-apidoc-") as directory:
        template = Path(directory) / "package.rst.jinja"
        template.write_text(summary, encoding="utf-8")
        arguments = [
            "--force",
            "--separate",
            "--no-toc",
            "--templatedir",
            directory,
            "--output-dir",
            str(output),
            str(package),
            *(str(path) for path in package.glob("*.py") if path.name != "__init__.py"),
        ]
        if apidoc(arguments):
            raise RuntimeError("Sphinx apidoc failed.")
    for name in optional:
        text = (
            name + "\n" + "=" * len(name) + "\n\n"
            ".. py:class:: ov_radiomics." + name + "\n\n"
            "   This public name is a lazy alias supplied by the optional Python\n"
            "   reference backend. Its implementation is intentionally not imported\n"
            "   during documentation generation. Install ``ov-radiomics[reference]``\n"
            "   only when using that documented legacy path.\n\n"
            "   Native pipelines return their typed native result objects instead.\n"
            "   See :doc:`../learning-native` and :doc:`../models-native`.\n"
        )
        (output / ("ov_radiomics." + name + ".rst")).write_text(text, encoding="utf-8")
    print(
        "Generated API for",
        len(normal) + len(optional),
        "public names;",
        len(optional),
        "optional aliases documented without imports.",
    )
    return normal, optional


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    build(arguments.root, arguments.output)
