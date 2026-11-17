"""Small exact-f64 scalar audit; freezes rational oracles before either backend runs.

No estimator, Golden, feature definition, package dependency or algorithm is changed.
The only oracle inputs are float.hex representations of actual binary64 values.
"""

import argparse
import hashlib
import json
import math
import platform
import struct
import subprocess
import sys
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path

FEATURES = ("Mean", "Variance", "Energy")
CONFIG = dict(
    profile="python_golden_v1",
    discretization=dict(method="fixed_count", count=8),
    aggregation="direction_mean",
    gldm_alpha=0,
    voxel_array_shift=0.0,
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def decimal_text(value):
    with localcontext() as context:
        context.prec = 80
        return str(Decimal(value.numerator) / Decimal(value.denominator))


def rational(value):
    return dict(
        numerator=str(value.numerator),
        denominator=str(value.denominator),
        decimal80=decimal_text(value),
    )


def fraction(item):
    return Fraction(int(item["numerator"]), int(item["denominator"]))


def exact_features(values):
    xs = [Fraction.from_float(value) for value in values]
    n = len(xs)
    mean = sum(xs, Fraction()) / n
    return dict(
        Mean=mean,
        Variance=sum(((x - mean) ** 2 for x in xs), Fraction()) / n,
        Energy=sum((x * x for x in xs), Fraction()),
    )


def freeze(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    cases = [
        (
            "ordinary",
            [0.1, 0.2, 0.3, -0.4, 1.5, 2.0, -1.0, 3.25],
            "Ordinary actual f64 values, not exact decimal tenths",
        ),
        (
            "cancellation",
            [1e16, 1.0, -1e16, 1.0, 1e16, 1.0, -1e16, 1.0],
            "Signed cancellation in the mean; input order preserved",
        ),
        (
            "large_offset_small_variance",
            [1e12 + i / 8192 for i in (0, 1, 2, 3, 4, 5, 6, 8)],
            (
                "Large exactly represented offset with 1/8192-spaced f64 values and"
                " nonrepresentable exact mean"
            ),
        ),
        ("constant", [7.0] * 8, "Exactly representable constant; population variance zero"),
    ]
    manifest = dict(
        schema="radiomics.scalar-accuracy-inputs.v1",
        features=list(FEATURES),
        oracle=(
            "Exact Fraction.from_float of each frozen float.hex input; Decimal80 is display only"
        ),
        formulas=dict(
            Mean="sum(x)/n", Variance="sum((x-exact_mean)^2)/n", Energy="sum(x*x), shift=0"
        ),
        settings=dict(
            native_config=CONFIG,
            python_binCount=8,
            python_voxelArrayShift=0,
            shape_yx=[2, 4],
            mask="all eight voxels selected as label 1",
            preprocessing="none",
            interpolation="none",
            filters="none",
            spatial_spacing=[1.0, 1.0],
        ),
        reference_precision=(
            "Exact rational result and correctly rounded binary64 reference; relative error"
            " undefined at exact zero"
        ),
        source_tool_sha256=sha256(__file__),
        oracle_python=sys.version,
        cases=[
            dict(
                id=name,
                purpose=purpose,
                input_f64_hex=[x.hex() for x in values],
                reference={
                    key: {**rational(value), "rounded_f64_hex": float(value).hex()}
                    for key, value in exact_features(values).items()
                },
            )
            for name, values, purpose in cases
        ],
    )
    path = output / "manifest.json"
    write_json(path, manifest)
    print("Frozen before any native/reference execution:", path, sha256(path))


def load_manifest(path, expected):
    if sha256(path) != expected:
        raise ValueError("Frozen manifest SHA256 differs.")
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest["source_tool_sha256"] != sha256(__file__):
        raise ValueError("Audit tool differs from the pre-execution frozen source.")
    for case in manifest["cases"]:
        computed = exact_features([float.fromhex(x) for x in case["input_f64_hex"]])
        if any(computed[key] != fraction(case["reference"][key]) for key in FEATURES):
            raise ValueError("Frozen oracle reconstruction differs.")
    return manifest


def worker(manifest_path, expected, backend, output):
    manifest = load_manifest(manifest_path, expected)
    import numpy as np

    identity = dict(
        python=sys.version,
        interpreter=sys.executable,
        numpy=np.__version__,
        platform=platform.platform(),
    )
    if backend == "native":
        import ov_radiomics as rad
        from ov_radiomics._backend import resolve_backend

        identity.update(
            package=rad.__version__,
            package_file=rad.__file__,
            backend=rad.backend_info(),
            native_extension_sha256=sha256(resolve_backend("rust").__file__),
        )
        if identity["backend"]["metrics_backend"] != "rust":
            raise RuntimeError("Native audit did not obtain the Rust backend.")
    else:
        import radiomics
        import SimpleITK as sitk
        from radiomics import firstorder

        if radiomics.__version__ != "v3.0.1":
            raise RuntimeError("Expected the unchanged public PyRadiomics v3.0.1 environment.")
        identity.update(
            pyradiomics=radiomics.__version__,
            simpleitk=sitk.Version_VersionString(),
            firstorder_source_sha256=sha256(firstorder.__file__),
            package_file=radiomics.__file__,
        )
    results = []
    for case in manifest["cases"]:
        values = np.asarray(
            [float.fromhex(x) for x in case["input_f64_hex"]], dtype=np.float64
        ).reshape(2, 4)
        # Verify exactly the same input binary64 values immediately before calls.
        if [float(x).hex() for x in values.ravel()] != case["input_f64_hex"]:
            raise RuntimeError("Backend array conversion changed actual input bits.")
        try:
            if backend == "native":
                image = rad.image.from_array(values, spatial_unit="mm")
                mask = rad.mask.from_array(
                    np.ones_like(values, dtype=np.uint32), geometry=image.geometry
                )
                sample = rad.ImageSample(
                    dict(sample_id=case["id"], patient_id=case["id"]), image, mask
                )
                table = sample.extract(CONFIG, families=["firstorder"])
                observed = {
                    key: float(table.features.iloc[0]["original_firstorder_" + key])
                    for key in FEATURES
                }
            else:
                image = sitk.GetImageFromArray(values)
                mask = sitk.GetImageFromArray(np.ones_like(values, dtype=np.uint8))
                extractor = firstorder.RadiomicsFirstOrder(
                    image, mask, label=1, binCount=8, voxelArrayShift=0
                )
                extractor.disableAllFeatures()
                for key in FEATURES:
                    extractor.enableFeatureByName(key)
                observed = {key: float(value) for key, value in extractor.execute().items()}
            results.append(
                dict(
                    case=case["id"],
                    status="ok",
                    input_f64_hex=case["input_f64_hex"],
                    values={
                        key: dict(value=observed[key], f64_hex=observed[key].hex())
                        for key in FEATURES
                    },
                )
            )
        except Exception as error:
            results.append(
                dict(
                    case=case["id"],
                    status="rejected",
                    error_type=type(error).__name__,
                    message=str(error),
                )
            )
    write_json(
        output,
        dict(
            schema="radiomics.scalar-accuracy-worker.v1",
            backend=backend,
            manifest_sha256=expected,
            source_tool_sha256=sha256(__file__),
            identity=identity,
            results=results,
        ),
    )


def ordered_bits(value):
    bits = struct.unpack(">Q", struct.pack(">d", value))[0]
    return (~bits & ((1 << 64) - 1)) if bits >> 63 else bits | (1 << 63)


def errors(actual, reference):
    if not math.isfinite(actual):
        return dict(status="nonfinite", f64_hex=actual.hex())
    exact = Fraction.from_float(actual)
    absolute = abs(exact - reference)
    rounded = float(reference)
    spacing = Fraction.from_float(math.ulp(rounded))
    return dict(
        status="finite",
        actual_f64_hex=actual.hex(),
        absolute_error=rational(absolute),
        relative_error=None if reference == 0 else rational(absolute / abs(reference)),
        relative_error_status="undefined_zero_reference" if reference == 0 else "defined",
        correctly_rounded_reference_f64_hex=rounded.hex(),
        ulp_distance_to_rounded_reference=abs(ordered_bits(actual) - ordered_bits(rounded)),
        error_in_reference_ulp=rational(absolute / spacing),
        exact_rational_match=absolute == 0,
    )


def run(manifest_path, expected, native_python, python301):
    manifest_path = Path(manifest_path).resolve()
    manifest = load_manifest(manifest_path, expected)
    output = manifest_path.parent
    raw = {}
    for backend, executable in (("native", native_python), ("pyradiomics301", python301)):
        target = output / (backend + ".json")
        command = [
            str(executable),
            str(Path(__file__).resolve()),
            "worker",
            "--manifest",
            str(manifest_path),
            "--manifest-sha256",
            expected,
            "--backend",
            backend,
            "--output",
            str(target),
        ]
        process = subprocess.run(
            command, capture_output=True, text=True, encoding="utf-8", errors="replace"
        )
        (output / (backend + ".stdout.txt")).write_text(process.stdout, encoding="utf-8")
        (output / (backend + ".stderr.txt")).write_text(process.stderr, encoding="utf-8")
        if process.returncode:
            raise RuntimeError("Backend worker failed; retained logs: " + backend)
        raw[backend] = json.loads(target.read_text(encoding="utf-8"))
    comparisons = []
    for case in manifest["cases"]:
        results = {
            name: next(x for x in data["results"] if x["case"] == case["id"])
            for name, data in raw.items()
        }
        if any(value["status"] != "ok" for value in results.values()):
            comparisons.append(
                dict(case=case["id"], status="not_comparable", backend_results=results)
            )
            continue
        for key in FEATURES:
            reference = fraction(case["reference"][key])
            actuals = {
                name: float.fromhex(data["values"][key]["f64_hex"])
                for name, data in results.items()
            }
            absolute = {
                name: abs(Fraction.from_float(value) - reference) for name, value in actuals.items()
            }
            winner = "tie" if len(set(absolute.values())) == 1 else min(absolute, key=absolute.get)
            comparisons.append(
                dict(
                    case=case["id"],
                    feature=key,
                    status="compared",
                    smaller_absolute_error=winner,
                    native_python_absolute_difference=decimal_text(
                        abs(
                            Fraction.from_float(actuals["native"])
                            - Fraction.from_float(actuals["pyradiomics301"])
                        )
                    ),
                    backends={name: errors(value, reference) for name, value in actuals.items()},
                )
            )
    report = dict(
        schema="radiomics.scalar-accuracy-report.v1",
        manifest_sha256=expected,
        source_tool_sha256=sha256(__file__),
        features=list(FEATURES),
        oracle=manifest["oracle"],
        comparisons=comparisons,
        worker_files_sha256={name: sha256(output / (name + ".json")) for name in raw},
        limitation=(
            "Four tiny input cases and three same-definition scalars only. No universal"
            " precision/clinical claim; no change to python_golden_v1 or its tolerances."
        ),
    )
    write_json(output / "report.json", report)
    print(
        json.dumps(
            {
                name: sum(x.get("smaller_absolute_error") == name for x in comparisons)
                for name in ("native", "pyradiomics301", "tie")
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze_parser = commands.add_parser("freeze")
    freeze_parser.add_argument("--output", type=Path, required=True)
    for name in ("worker", "run"):
        command = commands.add_parser(name)
        command.add_argument("--manifest", type=Path, required=True)
        command.add_argument("--manifest-sha256", required=True)
        if name == "worker":
            command.add_argument("--backend", choices=("native", "pyradiomics301"), required=True)
            command.add_argument("--output", type=Path, required=True)
        else:
            command.add_argument("--native-python", type=Path, required=True)
            command.add_argument("--python301", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        freeze(args.output)
    elif args.command == "worker":
        worker(args.manifest, args.manifest_sha256, args.backend, args.output)
    else:
        run(args.manifest, args.manifest_sha256, args.native_python, args.python301)


if __name__ == "__main__":
    main()
