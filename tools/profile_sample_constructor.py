"""Narrow matched-wheel constructor diagnostic, with no method monkeypatches.

Before/after: same immutable .pyd/core, frozen PAT1 CT66, warmup1 +7 samples
for complete L2 and constructor-only; +3 fresh-process RSS workers per wheel.
This is a diagnostic, not a replacement for the original854 observations.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

from profile_native_boundary import (
    ROOT,
    check_features,
    host_info,
    load_inputs,
    sha,
    summary,
    write_json,
)
from profile_native_memory import validate_worker

PLAN = dict(
    schema="radiomics.constructor-diagnostic-plan.v1",
    cpu_warmups=1,
    cpu_repeats=7,
    rss_repeats=3,
    cpu_stages=["full_l2", "sample_constructor"],
    native_threads=1,
    frozen_features=66,
    fixed_width=25,
    roi_label=1,
    profile="python_golden_v1",
    instrumented=False,
    description=(
        "Original public constructors/extract/DataFrame; no to_dict/request tracing or"
        " monkeypatching."
    ),
)


def installed_identity(wheel, build):
    import numpy
    import pandas
    import threadpoolctl

    import ov_radiomics as rad
    from ov_radiomics._backend import resolve_backend

    native = resolve_backend("rust")
    package = Path(rad.__file__).resolve().parent
    with zipfile.ZipFile(wheel) as archive:
        files = {
            name: hashlib.sha256(archive.read(name)).hexdigest()
            for name in archive.namelist()
            if name.startswith("ov_radiomics/") and name.endswith((".py", ".pyd", ".so"))
        }
    if not files:
        raise ValueError("No package payload in supplied wheel.")
    actual = {}
    for name, expected in files.items():
        path = package / Path(name).relative_to("ov_radiomics")
        actual[name] = sha(path)
        if actual[name] != expected:
            raise ValueError("Installed package differs from its frozen wheel: " + name)
    native_digest = sha(native.__file__)
    expected = build["wheel_files_sha256"]["ov_radiomics/_native.pyd"]
    if native_digest != expected:
        raise ValueError("Both environments must use exactly the same supplied v4 .pyd bytes.")
    if native.IMPLEMENTATION_SHA256 != build["core_implementation_sha256"]:
        raise ValueError("Core implementation differs from the supplied immutable build.")
    return dict(
        python=sys.executable,
        pid=os.getpid(),
        package_path=str(package),
        wheel=str(wheel),
        wheel_sha256=sha(wheel),
        installed_payload_sha256=actual,
        native_sha256=native_digest,
        core_implementation_sha256=native.IMPLEMENTATION_SHA256,
        dependencies={
            name: dict(
                version=module.__version__, path=module.__file__, sha256=sha(module.__file__)
            )
            for name, module in [
                ("numpy", numpy),
                ("pandas", pandas),
                ("threadpoolctl", threadpoolctl),
            ]
        },
    )


def worker(args):
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "RAYON_NUM_THREADS",
    ):
        os.environ[name] = "1"
    build = json.loads(args.build_manifest.read_text())
    identity = installed_identity(args.wheel, build)
    if args.worker == "identity":
        write_json(args.output, identity)
        return
    import pandas as pd
    import threadpoolctl

    import ov_radiomics as rad

    values, labels, spec, reference = load_inputs(args.fixtures, ROOT)
    config, affine = spec["native_extraction"], spec["dataset"]["prepared_affine_ras_mm"]
    metadata = dict(sample_id="public_ct_pat1-R0000", patient_id="P0000")
    if args.stage == "sample_constructor":
        image = rad.image.from_array(values, affine=affine, spatial_unit="mm")
        mask = rad.mask.from_array(labels, geometry=image.geometry)

    def call():
        if args.stage == "sample_constructor":
            return rad.ImageSample(metadata, image, mask)
        current_image = rad.image.from_array(values, affine=affine, spatial_unit="mm")
        current_mask = rad.mask.from_array(labels, geometry=current_image.geometry)
        sample = rad.ImageSample(metadata, current_image, current_mask)
        table = sample.extract(config)
        frame = pd.concat([table.features], ignore_index=True)
        return (
            frame[["ID"] + reference["output_feature_order"]]
            .sort_values("ID")
            .reset_index(drop=True)
        )

    raw = []
    with threadpoolctl.threadpool_limits(limits=1):
        pools = threadpoolctl.threadpool_info()
        if any(pool.get("num_threads") != 1 for pool in pools):
            raise RuntimeError("Thread limit not effective.")
        for repetition in range(PLAN["cpu_warmups"] + PLAN["cpu_repeats"]):
            started = time.perf_counter_ns()
            value = call()
            elapsed = time.perf_counter_ns() - started
            record = dict(
                repetition=repetition - PLAN["cpu_warmups"],
                warmup=repetition < PLAN["cpu_warmups"],
                nanoseconds=elapsed,
                stage=args.stage,
            )
            # Persist even if later correctness checks fail.
            with args.output.with_suffix(".raw.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")
            frame = value.extract(config).features if args.stage == "sample_constructor" else value
            features = frame.iloc[0].to_dict()
            features.pop("ID")
            check_features(features, reference)
            if not record["warmup"]:
                raw.append(record)
            del value, frame, features
    # Check the package again after observations, outside the measured clocks.
    after = installed_identity(args.wheel, build)
    if after["installed_payload_sha256"] != identity["installed_payload_sha256"]:
        raise ValueError("Installed package changed during the run.")
    write_json(
        args.output,
        dict(
            identity=identity,
            stage=args.stage,
            raw=raw,
            threadpools=pools,
            summary=summary([row["nanoseconds"] for row in raw]),
            all_66_features_match_frozen_ct=True,
        ),
    )


def run_child(command, directory):
    directory.mkdir(exist_ok=False)
    with (directory / "stdout.txt").open("wb") as out, (directory / "stderr.txt").open("wb") as err:
        process = subprocess.Popen(command, stdout=out, stderr=err)
        launcher_pid = process.pid
        try:
            code = process.wait(timeout=180)
        except subprocess.TimeoutExpired:
            if process.poll() is None:
                if os.name == "nt":
                    subprocess.run(
                        ["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True
                    )
                else:
                    process.kill()
                process.wait(timeout=15)
            code = 124
    return dict(
        command=command,
        launcher_pid=launcher_pid,
        exit_code=code,
        stdout_sha256=sha(directory / "stdout.txt"),
        stderr_sha256=sha(directory / "stderr.txt"),
    )


def coordinator(args):
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("Use a new constructor diagnostic directory.")
    if any((path / ".git").exists() for path in [output, *output.parents]):
        raise ValueError("Keep public CT diagnostic outputs outside Git.")
    if not args.acknowledge_license:
        raise PermissionError("Explicit source licence acknowledgement is required.")
    _, _, spec, reference = load_inputs(args.fixtures, ROOT)
    output.mkdir(parents=True, exist_ok=False)
    environments = {
        "baseline": (args.baseline_python, args.baseline_wheel),
        "candidate": (args.candidate_python, args.candidate_wheel),
    }
    write_json(
        output / "plan.json",
        dict(
            **PLAN,
            fixtures=str(args.fixtures.resolve()),
            build_manifest_sha256=sha(args.build_manifest),
            environments={
                name: dict(
                    python=str(pair[0].resolve()),
                    wheel=str(pair[1].resolve()),
                    wheel_sha256=sha(pair[1]),
                )
                for name, pair in environments.items()
            },
        ),
    )
    script = str(Path(__file__).resolve())
    common = [
        "--fixtures",
        str(args.fixtures.resolve()),
        "--build-manifest",
        str(args.build_manifest.resolve()),
    ]
    records = []
    identities = {}
    # Verify both wheels before any measurements. Separate workers ensure these
    # wheel hashes/import costs cannot become RSS setup for the measured worker.
    for name, (python, wheel) in environments.items():
        directory = output / ("identity-" + name)
        file = directory / "worker.json"
        command = [
            str(python.resolve()),
            script,
            "--worker",
            "identity",
            "--wheel",
            str(wheel.resolve()),
            "--output",
            str(file),
            *common,
        ]
        launch = run_child(command, directory)
        if launch["exit_code"] != 0:
            raise RuntimeError("Wheel preflight failed; see " + str(directory))
        identities[name] = json.loads(file.read_text())
    if (
        identities["baseline"]["native_sha256"] != identities["candidate"]["native_sha256"]
        or identities["baseline"]["core_implementation_sha256"]
        != identities["candidate"]["core_implementation_sha256"]
    ):
        raise ValueError("Before/after native binaries/core must match exactly.")
    dependency_ids = [
        {
            name: (record["version"], record["sha256"])
            for name, record in identity["dependencies"].items()
        }
        for identity in identities.values()
    ]
    if dependency_ids[0] != dependency_ids[1]:
        raise ValueError("Baseline/candidate numerical or thread-control dependencies differ.")
    # FullL2 has its own process, without extra constructor-only objects retained.
    jobs = [
        ("baseline", "full_l2"),
        ("candidate", "full_l2"),
        ("candidate", "sample_constructor"),
        ("baseline", "sample_constructor"),
    ]
    for name, stage in jobs:
        python, wheel = environments[name]
        directory = output / (name + "-" + stage)
        file = directory / "worker.json"
        command = [
            str(python.resolve()),
            script,
            "--worker",
            "timing",
            "--stage",
            stage,
            "--wheel",
            str(wheel.resolve()),
            "--output",
            str(file),
            *common,
        ]
        launch = run_child(command, directory)
        row = dict(environment=name, kind="timing", stage=stage, launch=launch, status="failed")
        if launch["exit_code"] == 0:
            result = json.loads(file.read_text())
            row.update(status="passed", result=result)
            if len(result["raw"]) != PLAN["cpu_repeats"]:
                raise ValueError("Missing timing observation.")
        records.append(row)
        write_json(directory / "observation.json", row)
    # Reuse the uninstrumented self-RSS public-L2 worker, never request_trace.
    for repetition in range(PLAN["rss_repeats"]):
        order = ["baseline", "candidate"] if repetition % 2 == 0 else ["candidate", "baseline"]
        for name in order:
            python, _ = environments[name]
            directory = output / ("rss-%02d-%s" % (repetition, name))
            file = directory / "worker.json"
            command = [
                str(python.resolve()),
                str(ROOT / "tools/profile_native_memory.py"),
                "--worker",
                "python_l2",
                "--fixtures",
                str(args.fixtures.resolve()),
                "--request",
                str(args.request.resolve()),
                "--build-manifest",
                str(args.build_manifest.resolve()),
                "--output",
                str(file),
            ]
            launch = run_child(command, directory)
            row = dict(
                environment=name, kind="rss", repetition=repetition, launch=launch, status="failed"
            )
            if launch["exit_code"] == 0:
                result = json.loads(file.read_text())
                footer = json.loads((directory / "stdout.txt").read_text().strip().splitlines()[-1])
                peak = validate_worker(result, footer)
                check_features(result["features"], reference)
                result.pop("features")
                row.update(
                    status="passed",
                    result=result,
                    footer=footer,
                    actual_worker_pid=result["pid"],
                    lifetime_peak_rss_bytes=peak,
                )
            records.append(row)
            write_json(directory / "observation.json", row)
    # Installed package identity cannot change during the measurement window.
    for name, (python, wheel) in environments.items():
        directory = output / ("identity-after-" + name)
        file = directory / "worker.json"
        command = [
            str(python.resolve()),
            script,
            "--worker",
            "identity",
            "--wheel",
            str(wheel.resolve()),
            "--output",
            str(file),
            *common,
        ]
        launch = run_child(command, directory)
        if (
            launch["exit_code"]
            or json.loads(file.read_text())["installed_payload_sha256"]
            != identities[name]["installed_payload_sha256"]
        ):
            raise ValueError("Installed package changed across measurement window.")
    report = dict(
        schema="radiomics.sample-constructor-diagnostic.v1",
        status="passed" if all(row["status"] == "passed" for row in records) else "failed",
        kind="diagnostic_not_formal_benchmark",
        plan=PLAN,
        host=host_info(),
        identities=identities,
        script_sha256=sha(__file__),
        helper_sha256={
            name: sha(ROOT / "tools" / name)
            for name in ("profile_native_boundary.py", "profile_native_memory.py")
        },
        raw=records,
        frozen_ct_reference_sha256=sha(args.fixtures / "bbox_crop_reference.json"),
        limitations=[
            "No monkeypatches. Custom override guards therefore remain active as in production.",
            (
                "Complete public L2 is measured independently of prevalidated constructor-only"
                " stage; do not subtract medians."
            ),
            (
                "RSS is actual-worker lifetime HWM, including common imports/cropped input setup;"
                " not exact resettable phase peak."
            ),
            (
                "Both environments use Rust. This is not a new Rust-versus-PyRadiomics speed or"
                " memory comparison."
            ),
            (
                "The original854 observations/targets are unchanged. No new production result is"
                " inferred from a prototype alone."
            ),
        ],
    )
    write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=["identity", "timing"])
    parser.add_argument("--stage", choices=PLAN["cpu_stages"])
    parser.add_argument("--wheel", type=Path)
    parser.add_argument("--baseline-python", type=Path)
    parser.add_argument("--candidate-python", type=Path)
    parser.add_argument("--baseline-wheel", type=Path)
    parser.add_argument("--candidate-wheel", type=Path)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--request", type=Path)
    parser.add_argument("--build-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--acknowledge-license", action="store_true")
    args = parser.parse_args()
    if args.worker:
        if not args.wheel or args.worker == "timing" and not args.stage:
            parser.error("Worker requires wheel and timing stage.")
        worker(args)
    else:
        if not all(
            (
                args.baseline_python,
                args.candidate_python,
                args.baseline_wheel,
                args.candidate_wheel,
                args.request,
            )
        ):
            parser.error("Coordinator requires both Python/wheel pairs and request path.")
        result = coordinator(args)
        print("Constructor diagnostic:", args.output / "report.json")
        return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
