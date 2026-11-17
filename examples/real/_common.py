"""Small input/output and display utilities; calculations stay in public Rust APIs."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    def encode(item):
        if isinstance(item, Path):
            return str(item)
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, np.ndarray):
            return item.tolist()
        raise TypeError(type(item).__name__)

    with Path(path).open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, allow_nan=False, indent=2, default=encode)
        handle.write("\n")


def output_directory(path):
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path


def read_rows(path, required):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not set(required) <= set(reader.fieldnames or []):
            raise ValueError(
                "Missing required CSV columns: "
                + ", ".join(sorted(set(required) - set(reader.fieldnames or [])))
            )
        rows = list(reader)
    if not rows:
        raise ValueError("Input manifest is empty.")
    for row in rows:
        if any(row.get(key) is None or not row[key].strip() for key in required):
            raise ValueError("Required IDs/paths must be nonempty strings.")
        for key, value in list(row.items()):
            if value is None:
                row[key] = ""
    return rows


def save_feature_table(table, path):
    table.save(path)
    table.metadata.to_csv(Path(path).with_name(Path(path).stem + "-metadata.csv"), index=False)


def plot_overlay(directory, path, label, coordinate_label):
    """Render only a bounded, already native-rasterized display canvas."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    canvas = manifest["canvas"]
    if canvas["width"] * canvas["height"] > 4_000_000:
        raise ValueError("Preview canvas exceeds4million pixels; increase --overlay-pixel-size.")
    values = np.full((canvas["height"], canvas["width"]), np.nan)
    coverage = np.zeros(values.shape, dtype=np.uint64)
    for block in manifest["blocks"]:
        with np.load(directory / block["path"]) as data:
            y, x, h, w = block["y"], block["x"], block["height"], block["width"]
            values[y : y + h, x : x + w] = data["values"]
            coverage[y : y + h, x : x + w] = data["coverage"]
    x0, y0 = canvas["origin0"]
    sx, sy = canvas["pixel_size0"]
    extent = (x0, x0 + canvas["width"] * sx, y0 + canvas["height"] * sy, y0)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad((0, 0, 0, 0))
    image = axes[0].imshow(
        np.ma.masked_invalid(values),
        origin="upper",
        extent=extent,
        cmap=cmap,
        interpolation="nearest",
    )
    fig.colorbar(image, ax=axes[0], label=label)
    image = axes[1].imshow(
        np.ma.masked_where(coverage == 0, coverage),
        origin="upper",
        extent=extent,
        cmap=cmap,
        interpolation="nearest",
        vmin=0,
        vmax=max(1, int(coverage.max())),
    )
    from matplotlib.ticker import MaxNLocator

    bar = fig.colorbar(image, ax=axes[1], label="Valid tile coverage count")
    bar.locator = MaxNLocator(integer=True)
    bar.update_ticks()
    axes[0].set_title("Actual measured tile feature; overlap mean")
    axes[1].set_title("Uncovered pixels transparent")
    for axis in axes:
        axis.set(xlabel=coordinate_label + " x", ylabel=coordinate_label + " y")
    fig.savefig(path, dpi=130, transparent=True)
    plt.close(fig)


def plot_patch_qc(patch, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    image = patch.image.array
    mask = np.asarray(patch.qc["tissue_mask"], dtype=bool).reshape(image.shape[:2])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].imshow(image)
    axes[0].set_title("Actual source image pixels")
    axes[1].imshow(image)
    axes[1].imshow(np.ma.masked_where(~mask, mask), cmap="spring", alpha=0.35, vmin=0, vmax=1)
    scores = patch.qc["scores"]
    axes[1].set_title(
        "Rust QC tissue threshold mask\nTissue fraction=" + format(scores["tissue_fraction"], ".4f")
    )
    for axis in axes:
        axis.set(xlabel="Decoded patch x (pixels)", ylabel="Decoded patch y (pixels)")
    fig.suptitle("QC display only; no automatic segmentation or clinical outcome.", fontsize=9)
    fig.savefig(path, dpi=130)
    plt.close(fig)
