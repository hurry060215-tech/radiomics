"""Execute synthetic lessons for a disposable documentation source tree.

Reuse the native tutorial executor and its hash, kernel, image and artifact
checks. Public downloads and private-data lessons keep their explicit opt-in
execution paths. No maintained notebook or historical receipt is overwritten.
"""

import argparse
import shutil
from pathlib import Path

from build_tutorial_notebooks import generate_notebooks
from run_tutorials import EXPECTED, run


def prepare(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    inputs = output / "inputs"
    generate_notebooks(root, inputs)
    receipt = run(inputs, output / "execution", root)
    if receipt["status"] != "passed":
        raise RuntimeError("Documentation tutorial execution failed; see execution/receipt.json")
    stage = output / "website"
    # Website download directives deliberately reference maintained ../docs.
    shutil.copytree(root / "docs", output / "docs")
    shutil.copytree(root / "website", stage, ignore=shutil.ignore_patterns("_build"))
    for name in EXPECTED:
        shutil.copy2(
            output / "execution" / name / (name + ".ipynb"), stage / "tutorials" / (name + ".ipynb")
        )
    return stage


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(prepare(args.root, args.output))
