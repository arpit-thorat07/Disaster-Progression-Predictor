from pathlib import Path
import csv
import math

import numpy as np
import rasterio
from rasterio.enums import Resampling

ROOT = Path(r"C:\Disaster Progression Dataset")
REPORTS = ROOT / "audit" / "reports"

INVENTORY = REPORTS / "step1_file_inventory.csv"
OUTPUT = REPORTS / "step2_raster_validation.csv"
SUMMARY = REPORTS / "step2_raster_validation_summary.txt"

DATASET_ROOTS = {
    "GFF": ROOT / "GFF",
    "SEN12FLOOD": ROOT / "SEN12FLOOD",
    "TS-SatFire": ROOT / "TS-SatFire-selected",
}

if not INVENTORY.exists():
    raise FileNotFoundError(
        f"Inventory not found: {INVENTORY}\n"
        "Run the Step 1 inventory script first."
    )

results = []
counts = {
    name: {
        "checked": 0,
        "read_ok": 0,
        "read_errors": 0,
        "missing_crs": 0,
        "invalid_dimensions": 0,
        "pixel_sample_errors": 0,
    }
    for name in DATASET_ROOTS
}

with INVENTORY.open(
    "r", newline="", encoding="utf-8-sig"
) as f:
    inventory = list(csv.DictReader(f))

tif_rows = [
    row for row in inventory
    if row.get("extension", "").lower() in (".tif", ".tiff")
]

total = len(tif_rows)
print(f"Raster files to inspect: {total}")

for index, row in enumerate(tif_rows, start=1):
    dataset = row["dataset"]
    base = DATASET_ROOTS[dataset]
    path = base / row["relative_path"]

    result = {
        "dataset": dataset,
        "relative_path": row["relative_path"],
        "filename": row["filename"],
        "size_bytes": row["size_bytes"],
        "status": "ERROR",
        "width": "",
        "height": "",
        "band_count": "",
        "crs": "",
        "bounds": "",
        "resolution": "",
        "dtypes": "",
        "nodata": "",
        "valid_sample_pct": "",
        "error": "",
    }

    counts[dataset]["checked"] += 1

    try:
        with rasterio.open(path) as src:
            result["status"] = "OPEN_OK"
            result["width"] = src.width
            result["height"] = src.height
            result["band_count"] = src.count
            result["crs"] = str(src.crs) if src.crs else ""
            result["bounds"] = (
                f"{src.bounds.left},{src.bounds.bottom},"
                f"{src.bounds.right},{src.bounds.top}"
            )
            result["resolution"] = (
                f"{src.res[0]},{src.res[1]}"
            )
            result["dtypes"] = ",".join(src.dtypes)
            result["nodata"] = (
                str(src.nodata) if src.nodata is not None else ""
            )

            counts[dataset]["read_ok"] += 1

            if src.crs is None:
                counts[dataset]["missing_crs"] += 1

            if src.width <= 0 or src.height <= 0 or src.count <= 0:
                result["status"] = "INVALID_DIMENSIONS"
                counts[dataset]["invalid_dimensions"] += 1
            else:
                # Read a small sample to test pixel readability.
                # This avoids loading a complete large raster.
                sample_h = min(64, src.height)
                sample_w = min(64, src.width)

                try:
                    sample = src.read(
                        1,
                        out_shape=(sample_h, sample_w),
                        masked=True,
                        resampling=Resampling.nearest,
                    )

                    values = sample.astype("float64").filled(np.nan)
                    valid_pct = (
                        100.0 * np.isfinite(values).mean()
                    )
                    result["valid_sample_pct"] = round(
                        valid_pct, 2
                    )

                    if valid_pct == 0:
                        result["status"] = "NO_VALID_SAMPLE_PIXELS"

                except Exception as exc:
                    result["status"] = "PIXEL_SAMPLE_ERROR"
                    result["error"] = str(exc)
                    counts[dataset]["pixel_sample_errors"] += 1

    except Exception as exc:
        result["status"] = "OPEN_ERROR"
        result["error"] = str(exc)
        counts[dataset]["read_errors"] += 1

    results.append(result)

    if index % 500 == 0 or index == total:
        print(f"Progress: {index}/{total}")

fields = [
    "dataset", "relative_path", "filename", "size_bytes",
    "status", "width", "height", "band_count", "crs",
    "bounds", "resolution", "dtypes", "nodata",
    "valid_sample_pct", "error",
]

with OUTPUT.open(
    "w", newline="", encoding="utf-8-sig"
) as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(results)

lines = [
    "STEP 2: RASTER VALIDATION SUMMARY",
    "=" * 45,
    f"Total TIFF files inspected: {total}",
    "",
]

for dataset, stats in counts.items():
    lines.extend([
        f"Dataset: {dataset}",
        f"  Files inspected: {stats['checked']}",
        f"  Opened successfully: {stats['read_ok']}",
        f"  Open errors: {stats['read_errors']}",
        f"  Missing CRS: {stats['missing_crs']}",
        f"  Invalid dimensions: {stats['invalid_dimensions']}",
        f"  Pixel sample errors: {stats['pixel_sample_errors']}",
        "",
    ])

status_counts = {}
for result in results:
    status = result["status"]
    status_counts[status] = status_counts.get(status, 0) + 1

lines.extend([
    "Overall status counts:",
    *[f"  {k}: {v}" for k, v in sorted(status_counts.items())],
    "",
    f"Detailed CSV: {OUTPUT}",
])

SUMMARY.write_text("\n".join(lines), encoding="utf-8")

print("\n" + "\n".join(lines))
print(f"\nSaved detailed report: {OUTPUT}")
print(f"Saved summary: {SUMMARY}")
