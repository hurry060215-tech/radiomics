"""Synthetic 3D NIfTI/ROI -> native Python API and optional Python-free CLI.

python examples/native/01_extract.py --output /fresh/output --cli /absolute/radiomics
The CLI child receives an empty executable search path. Synthetic intensities
and bin width are a software demonstration, not a CT/MR/PET preset.
"""

import argparse
import csv
import json
import os
import struct
import subprocess
from pathlib import Path

import pandas as pd

import ov_radiomics as rad


def write_nifti(path, values):
    header = bytearray(352)
    struct.pack_into("<i", header, 0, 348)
    struct.pack_into("<8h", header, 40, 3, 6, 5, 4, 1, 1, 1, 1)
    struct.pack_into("<hh", header, 70, 64, 64)
    struct.pack_into("<8f", header, 76, 1, 2, 3, 4, 1, 1, 1, 1)
    struct.pack_into("<ff", header, 108, 352, 1)
    header[123] = 2
    struct.pack_into("<h", header, 254, 1)
    struct.pack_into("<4f", header, 280, -2, 0, 0, 10)
    struct.pack_into("<4f", header, 296, 0, 3, 0, -20)
    struct.pack_into("<4f", header, 312, 0, 0, 4, 30)
    header[344:348] = b"n+1\0"
    path.write_bytes(header + struct.pack("<" + "d" * len(values), *values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cli", type=Path)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / "inputs").mkdir()
    records = []
    coordinates = [(x, y, z) for z in range(4) for y in range(5) for x in range(6)]
    for index in range(3):
        sample = f"S{index + 1:03}"
        write_nifti(
            root / f"inputs/{sample}.nii", [x + 2 * y + 3 * z + index for x, y, z in coordinates]
        )
        write_nifti(
            root / f"inputs/{sample}_roi.nii",
            [int(1 <= x <= 4 and 1 <= y <= 3 and 1 <= z <= 2) for x, y, z in coordinates],
        )
        records.append(
            dict(
                sample_id=sample,
                patient_id=f"P{1 if index < 2 else 2:03}",
                image=f"inputs/{sample}.nii",
                mask=f"inputs/{sample}_roi.nii",
                modality="synthetic",
                region="tumor",
            )
        )
    cohort = root / "cohort.csv"
    with cohort.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    config = dict(
        extraction=dict(
            profile="python_golden_v1",
            discretization=dict(method="fixed_width", width=2.0),
            aggregation="direction_mean",
            gldm_alpha=0,
            voxel_array_shift=0.0,
        ),
        labels=[1],
        preparation=[],
    )
    (root / "extract.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    result = rad.extract(
        rad.read_cohort(cohort), params=config, backend="rust", profile="python_golden_v1"
    )
    result.save(root / "python-api")
    assert result.features.original_shape_basic_VoxelCount.eq(24).all()
    assert result.features.original_shape_basic_Volume_mm3.eq(576).all()
    if args.cli:
        binary = args.cli.resolve()
        (root / "empty-bin").mkdir()
        environment = {
            key: os.environ[key] for key in ("SystemRoot", "WINDIR") if key in os.environ
        }
        environment.update(PATH=str(root / "empty-bin"), TEMP=str(root), TMP=str(root))
        subprocess.run(
            [str(binary), "check", "--cohort", str(cohort), "--output", str(root / "check.json")],
            check=True,
            env=environment,
        )
        subprocess.run(
            [
                str(binary),
                "extract",
                "--cohort",
                str(cohort),
                "--config",
                str(root / "extract.json"),
                "--output",
                str(root / "rust-cli"),
                "--n-jobs",
                "2",
            ],
            check=True,
            env=environment,
        )
        from_cli = pd.read_csv(root / "rust-cli/features.csv", float_precision="round_trip").rename(
            columns={"sample_id": "ID"}
        )
        pd.testing.assert_frame_equal(
            from_cli, result.features, check_dtype=False, check_exact=True
        )
        cli = json.loads((root / "rust-cli/extraction.json").read_text())
        if cli["implementation_sha256"] != result.config["implementation_sha256"]:
            raise RuntimeError(
                "Rebuild CLI and wheel from the same core snapshot before claiming interoperability"
                " acceptance."
            )
    print(
        result.features[
            [
                "ID",
                "original_firstorder_Mean",
                "original_shape_basic_VoxelCount",
                "original_shape_basic_Volume_mm3",
            ]
        ].to_string(index=False)
    )
    print("Outputs:", root)


if __name__ == "__main__":
    main()
