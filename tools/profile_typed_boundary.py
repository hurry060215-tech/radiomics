"""Same-call five-stage boundary diagnostic, separate from frozen benchmark v2.

The installed package is measured without replacing to_dict methods, so the
canonical serializer guards execute exactly as in an uninstrumented call.
"""

import argparse
import json
import os
import statistics
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

from profile_native_boundary import check_features, host_info, load_inputs, sha, summary

ROOT = Path(__file__).resolve().parents[1]
STAGES = (
    "validate_image",
    "validate_mask",
    "validate_image_sample",
    "array_features",
    "validate_feature_table",
)


@contextmanager
def trace_boundary(common, types, native):
    """Observe encoding/decoding/copying in their actual enclosing phase."""
    original_encode = common.encode_json
    original_loads = json.loads
    original_copy = types.deepcopy
    owner = getattr(native, "OwnedArraySample", None)
    records, phases = [], []
    current = [None]

    def observed(kind, function, *args, **kwargs):
        start = time.perf_counter_ns()
        result = function(*args, **kwargs)
        records.append(
            dict(stage=current[0], kind=kind, nanoseconds=time.perf_counter_ns() - start)
        )
        return result

    def encode(*args, **kwargs):
        return observed("encode", original_encode, *args, **kwargs)

    def decode(*args, **kwargs):
        return observed("decode", original_loads, *args, **kwargs)

    def copy(*args, **kwargs):
        return observed("deepcopy", original_copy, *args, **kwargs)

    class OwnerProxy:
        def __init__(self, *args, **kwargs):
            self.owner = observed("native_owned_constructor", owner, *args, **kwargs)

        def __getattr__(self, name):
            return getattr(self.owner, name)

        def extract(self, *args, **kwargs):
            return observed("native_owned_extract", self.owner.extract, *args, **kwargs)

    @contextmanager
    def stage(name):
        previous, current[0] = current[0], name
        started = time.perf_counter_ns()
        try:
            yield
        finally:
            phases.append(dict(stage=name, nanoseconds=time.perf_counter_ns() - started))
            current[0] = previous

    common.encode_json, json.loads, types.deepcopy = encode, decode, copy
    if owner is not None:
        native.OwnedArraySample = OwnerProxy
    try:
        yield stage, records, phases
    finally:
        common.encode_json, json.loads, types.deepcopy = (
            original_encode,
            original_loads,
            original_copy,
        )
        if owner is not None:
            native.OwnedArraySample = owner


def run(args):
    if not args.acknowledge_license:
        raise PermissionError("Use --acknowledge-license for the existing licensed fixture.")
    output = args.output.resolve()
    if any((path / ".git").exists() for path in (output, *output.parents)):
        raise ValueError("Licensed inputs and raw diagnostic output must remain outside Git.")
    output.mkdir(parents=True, exist_ok=False)
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "RAYON_NUM_THREADS",
    ):
        os.environ[name] = "1"
    import numpy as np
    import pandas as pd
    import threadpoolctl

    import ov_radiomics as rad
    import ov_radiomics._native_common as common
    import ov_radiomics._types as types
    from ov_radiomics._backend import resolve_backend

    native = resolve_backend("rust")
    values, labels, spec, reference = load_inputs(args.fixtures.resolve(), ROOT)
    config = spec["native_extraction"]
    affine = spec["dataset"]["prepared_affine_ras_mm"]
    metadata = dict(sample_id="public_ct_pat1-R0000", patient_id="P0000")
    raw = []

    def extract(instrumented):
        if instrumented:
            with trace_boundary(common, types, native) as (stage, records, phases):
                start = time.perf_counter_ns()
                with stage("validate_image"):
                    image = rad.image.from_array(values, affine=affine, spatial_unit="mm")
                with stage("validate_mask"):
                    mask = rad.mask.from_array(labels, geometry=image.geometry)
                with stage("validate_image_sample"):
                    sample = rad.ImageSample(metadata, image, mask)
                # FeatureTable construction is nested inside the public extract
                # call; wrap __init__, without changing to_dict identity.
                original_init = types.FeatureTable.__init__

                def table_init(self, *args, **kwargs):
                    with stage("validate_feature_table"):
                        original_init(self, *args, **kwargs)

                types.FeatureTable.__init__ = table_init
                try:
                    with stage("array_features"):
                        table = sample.extract(config)
                finally:
                    types.FeatureTable.__init__ = original_init
                elapsed = time.perf_counter_ns() - start
            # Remove the nested table interval from array_features' inclusive
            # time only for the exclusive five-stage display. Both are retained.
            table_ns = sum(
                p["nanoseconds"] for p in phases if p["stage"] == "validate_feature_table"
            )
            exclusive = [
                {
                    **p,
                    "exclusive_nanoseconds": (
                        p["nanoseconds"] - (table_ns if p["stage"] == "array_features" else 0)
                    ),
                }
                for p in phases
            ]
        else:
            start = time.perf_counter_ns()
            image = rad.image.from_array(values, affine=affine, spatial_unit="mm")
            mask = rad.mask.from_array(labels, geometry=image.geometry)
            sample = rad.ImageSample(metadata, image, mask)
            table = sample.extract(config)
            elapsed = time.perf_counter_ns() - start
            records, exclusive = [], []
        frame = table.features.iloc[0].to_dict()
        frame.pop("ID")
        check_features(frame, reference)
        return dict(
            nanoseconds=elapsed,
            events=records,
            stages=exclusive,
            uses_owned_buffer=getattr(sample, "_owned_arrays", None) is not None,
        )

    with threadpoolctl.threadpool_limits(limits=1):
        for repetition in range(-args.warmups, args.repeats):
            for instrumented in (False, True):
                record = dict(
                    round=repetition,
                    warmup=repetition < 0,
                    instrumented=instrumented,
                    **extract(instrumented),
                )
                with (output / "raw.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record) + "\n")
                raw.append(record)
        pools = threadpoolctl.threadpool_info()
    selected = [r for r in raw if not r["warmup"] and r["instrumented"]]
    shares = [
        sum(e["nanoseconds"] for e in r["events"] if e["kind"] in ("encode", "decode", "deepcopy"))
        / r["nanoseconds"]
        for r in selected
    ]
    report = dict(
        schema="radiomics.typed-boundary-diagnostic.v1",
        kind="diagnostic_not_formal_benchmark",
        host=host_info(),
        source_revision=subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        native_sha256=sha(native.__file__),
        implementation_sha256=native.IMPLEMENTATION_SHA256,
        installed_package=rad.__file__,
        script_sha256=sha(__file__),
        fixture_reference_sha256=sha(args.fixtures / "bbox_crop_reference.json"),
        dependencies=dict(
            numpy=np.__version__, pandas=pd.__version__, threadpoolctl=threadpoolctl.__version__
        ),
        threadpools=pools,
        repeats=args.repeats,
        warmups=args.warmups,
        feature_count=66,
        all_features_match_frozen_reference=True,
        owned_buffer_used=all(r["uses_owned_buffer"] for r in selected),
        encode_decode_deepcopy_share_median=statistics.median(shares),
        instrumented_l2=summary([r["nanoseconds"] for r in selected]),
        uninstrumented_l2=summary(
            [r["nanoseconds"] for r in raw if not r["warmup"] and not r["instrumented"]]
        ),
        stages={
            name: summary(
                [
                    p["exclusive_nanoseconds"]
                    for r in selected
                    for p in r["stages"]
                    if p["stage"] == name
                ]
            )
            for name in STAGES
        },
        event_median_ms={
            name: (
                statistics.median(
                    [
                        sum(e["nanoseconds"] for e in r["events"] if e["kind"] == name)
                        for r in selected
                    ]
                )
                / 1e6
            )
            for name in (
                "encode",
                "decode",
                "deepcopy",
                "native_owned_constructor",
                "native_owned_extract",
            )
        },
        raw=raw,
        limitations=[
            (
                "The five stage display retains inclusive and exclusive same-call clocks; feature"
                " table validation is nested inside sample.extract."
            ),
            (
                "No to_dict method is replaced; the production canonical serializer guard remains"
                " active."
            ),
            (
                "The historical 52.59 percent is an earlier snapshot diagnostic, not a newly"
                " reproduced formal speed result."
            ),
            (
                "Original 854 benchmark observations, Golden and published L2/L3 values remain"
                " unchanged."
            ),
            "No image pixels, masks or request buffers are written to this report.",
        ],
    )
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "owned_buffer_used",
                    "encode_decode_deepcopy_share_median",
                    "instrumented_l2",
                    "uninstrumented_l2",
                    "stages",
                )
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--acknowledge-license", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1 or args.warmups < 0:
        parser.error("repeats must be positive and warmups nonnegative")
    run(args)


if __name__ == "__main__":
    main()
