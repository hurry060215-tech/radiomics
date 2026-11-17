"""Small fresh-process PAT1 memory diagnosis; not an official memory benchmark.

Workers report their own actual PID/current RSS/lifetime HWM. No Python launcher
PID is treated as the Rust worker, and no full source volume is loaded first.
"""

import argparse
import ctypes
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

from profile_native_boundary import ROOT, check_features, host_info, load_inputs, sha, write_json


def own_memory(stage):
    if os.name == "nt":

        class Counters(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("faults", ctypes.c_ulong)] + [
                (name, ctypes.c_size_t)
                for name in (
                    "peak_working_set",
                    "working_set",
                    "peak_paged",
                    "paged",
                    "peak_nonpaged",
                    "nonpaged",
                    "pagefile",
                    "peak_pagefile",
                )
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        ctypes.windll.kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        process = ctypes.windll.kernel32.GetCurrentProcess()
        function = ctypes.windll.psapi.GetProcessMemoryInfo
        function.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong]
        if not function(process, ctypes.byref(counters), counters.cb):
            raise OSError("Own process GetProcessMemoryInfo failed.")
        rss, peak = counters.working_set, counters.peak_working_set
        source = "own GetProcessMemoryInfo"
    elif sys.platform.startswith("linux"):
        lines = Path("/proc/self/status").read_text().splitlines()
        fields = {
            line.split(":")[0]: int(line.split()[1]) * 1024
            for line in lines
            if line.startswith(("VmRSS:", "VmHWM:"))
        }
        rss, peak = fields["VmRSS"], fields["VmHWM"]
        source = "own /proc/self/status"
    else:
        raise RuntimeError("Memory diagnostic supports Windows/Linux.")
    return dict(
        stage=stage, pid=os.getpid(), rss_bytes=rss, lifetime_peak_rss_bytes=peak, source=source
    )


def python_worker(args):
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "RAYON_NUM_THREADS",
    ):
        os.environ[key] = "1"
    import numpy as np
    import pandas as pd
    import threadpoolctl

    import ov_radiomics as rad
    from ov_radiomics._backend import resolve_backend

    native = resolve_backend("rust")
    build = json.loads(args.build_manifest.read_text())
    if sha(native.__file__) != build["wheel_files_sha256"]["ov_radiomics/_native.pyd"]:
        raise ValueError("Memory baseline must use the immutable v2 extension.")
    snapshots = [own_memory("common_python_imports_loaded")]
    # Only the tight-crop files: no full-source NIfTI decode or mask prior to HWM.
    values = np.load(args.fixtures / "values.npy", allow_pickle=False)
    labels = np.load(args.fixtures / "mask.npy", allow_pickle=False)
    reference = json.loads((args.fixtures / "bbox_crop_reference.json").read_text())
    spec = json.loads((ROOT / "benchmarks/v2/radiomics-ct-plan.json").read_text())
    snapshots.append(own_memory("cropped_arrays_loaded"))
    shallow_lists = None
    with threadpoolctl.threadpool_limits(limits=1):
        pools = threadpoolctl.threadpool_info()
        if args.worker == "python_l2":
            image = rad.image.from_array(
                values, affine=spec["dataset"]["prepared_affine_ras_mm"], spatial_unit="mm"
            )
            snapshots.append(own_memory("image_constructed"))
            mask = rad.mask.from_array(labels, geometry=image.geometry)
            snapshots.append(own_memory("mask_constructed"))
            sample = rad.ImageSample(
                dict(sample_id="public_ct_pat1-R0000", patient_id="P0000"), image, mask
            )
            snapshots.append(own_memory("sample_constructed"))
            table = sample.extract(spec["native_extraction"])
            frame = pd.concat([table.features], ignore_index=True)
            frame = (
                frame[["ID"] + reference["output_feature_order"]]
                .sort_values("ID")
                .reset_index(drop=True)
            )
            snapshots.append(own_memory("l2_complete_objects_retained"))
            features = frame.iloc[0].to_dict()
            features.pop("ID")
            buffers = [
                image._payload["values"],
                mask._payload["labels"],
                sample._payload["image"]["values"],
                sample._payload["mask"]["labels"],
            ]
            shallow_lists = dict(
                distinct_list_objects=len({id(buf) for buf in buffers}),
                elements_per_list=[len(buf) for buf in buffers],
                shallow_list_bytes=sum(sys.getsizeof(buf) for buf in buffers),
                excludes=(
                    "Boxed scalar objects, allocator overhead, temporary JSON and core arrays; only"
                    " a retained lower bound."
                ),
            )
        elif args.worker == "python_preencoded":
            text = args.request.read_text(encoding="utf-8")
            snapshots.append(own_memory("cropped_arrays_and_request_text_loaded"))
            response = native.request_json(text)
            features = json.loads(response)
            snapshots.append(own_memory("preencoded_gateway_complete"))
        else:
            raise ValueError("Unknown worker.")
    check_features(features, reference)
    result = dict(
        schema="radiomics.memory-worker.v1",
        path=args.worker,
        pid=os.getpid(),
        snapshots=snapshots,
        python=sys.executable,
        native_path=native.__file__,
        native_sha256=sha(native.__file__),
        loaded_numeric_bytes=int(values.nbytes + labels.nbytes),
        full_source_nifti_loaded=False,
        all_66_features_match_frozen_ct=True,
        features=features,
        threadpools=pools,
        retained_shallow_lists=shallow_lists,
    )
    write_json(args.output, result)
    # The final tiny footer is after output writing, from the actual worker.
    print(json.dumps(own_memory("after_output_written")), flush=True)


def validate_worker(worker, footer):
    actual_pid = worker.get("pid")
    if (
        not isinstance(actual_pid, int)
        or actual_pid <= 0
        or actual_pid == os.getpid()
        or footer.get("pid") != actual_pid
    ):
        raise ValueError(
            "PID mismatch: RSS must belong to the actual measured child, never its coordinator."
        )
    if any(snapshot["pid"] != actual_pid for snapshot in worker["snapshots"]):
        raise ValueError("A phase snapshot came from a different process.")
    if worker["full_source_nifti_loaded"] is not False:
        raise ValueError("Full-volume setup would obscure this preloaded L2 diagnostic.")
    snapshots = [*worker["snapshots"], footer]
    if any(s["rss_bytes"] < 0 or s["lifetime_peak_rss_bytes"] < s["rss_bytes"] for s in snapshots):
        raise ValueError("Invalid own-process RSS/HWM values.")
    return max(s["lifetime_peak_rss_bytes"] for s in snapshots)


def run(args):
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("Use a new memory diagnostic directory.")
    if any((p / ".git").exists() for p in [output, *output.parents]):
        raise ValueError("Keep licensed diagnostic data outside Git.")
    if not args.acknowledge_license:
        raise PermissionError("Explicit source licence acknowledgement is required.")
    # Coordinator-only verification; no image decoding precedes worker memory.
    values, labels, spec, reference = load_inputs(args.fixtures, ROOT)
    build = json.loads(args.build_manifest.read_text())
    payload = json.loads(args.request.read_text(encoding="utf-8"))["request"]
    import numpy as np

    if not np.array_equal(payload["image"]["values"], values.ravel()) or not np.array_equal(
        payload["morphology_mask"]["labels"], labels.ravel()
    ):
        raise ValueError("Typed JSON and cropped .npy workers would receive different buffers.")
    if (
        payload["image"]["geometry"]["affine"] != spec["dataset"]["prepared_affine_ras_mm"]
        or payload["config"] != spec["native_extraction"]
    ):
        raise ValueError("Typed JSON geometry/config differs from the fixed cropped CT.")
    output.mkdir(parents=True, exist_ok=False)
    modes = ["python_l2", "python_preencoded", "rust_core"]
    observations = []
    for round in range(args.repeats):
        for offset in range(len(modes)):
            mode = modes[(offset + round) % len(modes)]
            directory = output / ("%02d-%s" % (round, mode))
            directory.mkdir()
            result_path = directory / "worker.json"
            if mode == "rust_core":
                command = [
                    str(args.rust_probe.resolve()),
                    "--memory-preloaded",
                    str(args.request.resolve()),
                    str(result_path),
                ]
            else:
                command = [
                    str(args.native_python.resolve()),
                    str(Path(__file__).resolve()),
                    "--worker",
                    mode,
                    "--fixtures",
                    str(args.fixtures.resolve()),
                    "--request",
                    str(args.request.resolve()),
                    "--build-manifest",
                    str(args.build_manifest.resolve()),
                    "--output",
                    str(result_path),
                ]
            with (
                (directory / "stdout.txt").open("wb") as stdout,
                (directory / "stderr.txt").open("wb") as stderr,
            ):
                process = subprocess.Popen(command, stdout=stdout, stderr=stderr)
                child_pid = process.pid
                try:
                    code = process.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    if process.poll() is None:
                        if os.name == "nt":
                            subprocess.run(
                                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                capture_output=True,
                            )
                        else:
                            process.kill()
                        process.wait(timeout=15)
                    code = 124
            observation = dict(
                round=round, path=mode, launcher_pid=child_pid, exit_code=code, command=command
            )
            if code:
                observation["status"] = "failed"
                observations.append(observation)
                write_json(directory / "observation.json", observation)
                raise RuntimeError("Memory worker failed; logs and failed observation retained.")
            worker = json.loads(result_path.read_text())
            footer = json.loads((directory / "stdout.txt").read_text().strip().splitlines()[-1])
            peak = validate_worker(worker, footer)
            check_features(worker["features"], reference)
            if (
                mode == "rust_core"
                and worker["implementation_sha256"] != build["core_implementation_sha256"]
            ):
                raise ValueError(
                    "Memory Rust probe must use the same frozen feature implementation."
                )
            # Do not duplicate derived feature values in the timing/identity archive.
            worker.pop("features")
            pre_phase = (
                "preloaded_before_core"
                if mode == "rust_core"
                else (
                    "cropped_arrays_and_request_text_loaded"
                    if mode == "python_preencoded"
                    else "cropped_arrays_loaded"
                )
            )
            loaded = next(s for s in worker["snapshots"] if s["stage"] == pre_phase)
            observation.update(
                status="passed",
                worker=worker,
                worker_footer=footer,
                worker_pid=worker["pid"],
                lifetime_peak_rss_bytes=peak,
                loaded_rss_bytes=loaded["rss_bytes"],
                loaded_hwm_bytes=loaded["lifetime_peak_rss_bytes"],
                peak_minus_loaded_rss_bytes=peak - loaded["rss_bytes"],
                new_hwm_after_loaded_bytes=max(0, peak - loaded["lifetime_peak_rss_bytes"]),
            )
            observations.append(observation)
            write_json(directory / "observation.json", observation)
            with (output / "raw.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(observation) + "\n")
    results = {}
    for mode in modes:
        rows = [row for row in observations if row["path"] == mode]
        results[mode] = {
            key: statistics.median(row[key] for row in rows)
            for key in (
                "lifetime_peak_rss_bytes",
                "loaded_rss_bytes",
                "loaded_hwm_bytes",
                "peak_minus_loaded_rss_bytes",
                "new_hwm_after_loaded_bytes",
            )
        }
        results[mode]["n"] = len(rows)
    report = dict(
        schema="radiomics.native-memory-diagnosis.v1",
        kind="diagnostic_not_formal_memory_benchmark",
        host=host_info(),
        repeats=args.repeats,
        raw=observations,
        summaries=results,
        native_build_manifest_sha256=sha(args.build_manifest),
        native_extension_sha256=build["wheel_files_sha256"]["ov_radiomics/_native.pyd"],
        rust_probe_binary_sha256=sha(args.rust_probe),
        request_sha256=sha(args.request),
        prepared_image_f64_sha256=reference["image_f64_le_sha256"],
        prepared_mask_u32_sha256=reference["mask_u32_le_sha256"],
        helper_sha256={
            name: sha(ROOT / "tools" / name)
            for name in ("profile_native_memory.py", "profile_native_boundary.py")
        },
        all_66_features_match_frozen_ct=True,
        full_source_nifti_loaded_in_workers=False,
        limitations=[
            (
                "Own-process Windows PeakWorkingSetSize is lifetime HWM, not a resettable"
                " feature-phase peak. Setup cannot be subtracted to obtain an exact phase peak."
            ),
            (
                "loaded RSS and previous HWM are explicit; positive new_hwm_after_loaded proves the"
                " phase exceeded earlier lifetime HWM, not total allocation volume."
            ),
            (
                "python_l2 and python_preencoded share imports/cropped arrays; preencoded"
                " additionally retains UTF-8 request text and omits the full public object/table"
                " contract."
            ),
            (
                "Pure Rust has different startup cost and parses typed JSON, not .npy. Its"
                " standalone absolute peak cannot establish the memory of a Python-facing public"
                " API."
            ),
            (
                "No OS cache purge, no Linux execution in this run, no allocator instrumentation,"
                " no memory-budget enforcement or general maximum-memory claim."
            ),
        ],
    )
    write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=["python_l2", "python_preencoded"])
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--build-manifest", type=Path, required=True)
    parser.add_argument("--rust-probe", type=Path)
    parser.add_argument("--native-python", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--acknowledge-license", action="store_true")
    args = parser.parse_args()
    if args.worker:
        python_worker(args)
    else:
        if not args.rust_probe or not args.native_python or args.repeats < 1:
            parser.error("Coordinator requires rust-probe, native-python and positive repeats.")
        run(args)
        print("Memory diagnostic:", args.output / "report.json")


if __name__ == "__main__":
    main()
