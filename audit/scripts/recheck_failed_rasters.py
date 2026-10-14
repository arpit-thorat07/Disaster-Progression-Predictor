
import csv
from pathlib import Path

import rasterio

ROOT = Path(r"C:\Disaster Progression Dataset")
ERROR_CSV = ROOT / "audit" / "reports" / "step2_raster_errors.csv"
OUTPUT_CSV = ROOT / "audit" / "reports" / "step2_raster_recheck.csv"

with ERROR_CSV.open(
    "r", newline="", encoding="utf-8-sig"
) as f:
    rows = list(csv.DictReader(f))

results = []
total = len(rows)
print(f"Rechecking {total} flagged rasters...")

for i, row in enumerate(rows, start=1):
    path = ROOT / "SEN12FLOOD" / row["relative_path"]

    result = {
        "relative_path": row["relative_path"],
        "file_exists": path.is_file(),
        "file_size_bytes": "",
        "metadata_opened": False,
        "pixel_read_ok": False,
        "error": "",
    }

    try:
        if not path.is_file():
            raise FileNotFoundError("File does not exist")

        result["file_size_bytes"] = path.stat().st_size

        with rasterio.open(path) as src:
            result["metadata_opened"] = True

            src.read(
                1,
                window=(
                    (0, min(64, src.height)),
                    (0, min(64, src.width)),
                ),
            )

            result["pixel_read_ok"] = True

    except Exception as exc:
        result["error"] = str(exc)

    results.append(result)

    if i % 10 == 0 or i == total:
        print(f"Progress: {i}/{total}")

fields = [
    "relative_path",
    "file_exists",
    "file_size_bytes",
    "metadata_opened",
    "pixel_read_ok",
    "error",
]

with OUTPUT_CSV.open(
    "w", newline="", encoding="utf-8-sig"
) as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(results)

readable = sum(r["pixel_read_ok"] for r in results)
failed = total - readable

print(f"\nReadable: {readable}/{total}")
print(f"Still failing: {failed}")
print(f"Report saved: {OUTPUT_CSV}")

for result in results:
    if not result["pixel_read_ok"]:
        print(
            f"\nFAILED: {result['relative_path']}"
            f"\nError: {result['error']}"
        )
