import csv
import os
import rasterio
from rasterio.errors import RasterioError

ROOT = os.getcwd()
INPUT = os.path.join(ROOT, "audit", "reports", "step2_raster_recheck.csv")
OUTPUT = os.path.join(ROOT, "audit", "reports", "step3_tiff_diagnostics.csv")

results = []

with open(INPUT, newline="", encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        path = os.path.join(ROOT, "SEN12FLOOD", row["relative_path"])
        result = {
            "relative_path": row["relative_path"],
            "file_size_bytes": os.path.getsize(path) if os.path.exists(path) else None,
            "exists": os.path.exists(path),
            "width": None,
            "height": None,
            "bands": None,
            "dtype": None,
            "crs": None,
            "block_shapes": None,
            "read_error": None,
        }

        try:
            with rasterio.open(path) as src:
                result["width"] = src.width
                result["height"] = src.height
                result["bands"] = src.count
                result["dtype"] = ",".join(src.dtypes)
                result["crs"] = str(src.crs)
                result["block_shapes"] = str(src.block_shapes)

                try:
                    src.read(1, window=((0, min(64, src.height)),
                                        (0, min(64, src.width))))
                    result["read_error"] = "NONE"
                except Exception as e:
                    result["read_error"] = repr(e)

        except Exception as e:
            result["read_error"] = "OPEN ERROR: " + repr(e)

        results.append(result)
        print(f'{len(results):2}/{36}: {result["relative_path"]}')

with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print(f"\nSaved: {OUTPUT}")
print(f"Files diagnosed: {len(results)}")
print("Read failures:", sum(r["read_error"] != "NONE" for r in results))
