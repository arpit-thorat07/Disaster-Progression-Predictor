import csv
import json
from pathlib import Path
from collections import Counter

import numpy as np
import rasterio

ROOT = Path("GFF/base/rois")
OUT = Path("audit/reports")
OUT.mkdir(parents=True, exist_ok=True)

rows = []

for tif in sorted(ROOT.glob("*.tif")):
    meta_path = tif.with_name(tif.stem + "-meta.json")

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"Metadata error: {tif.name}: {exc}")
        continue

    try:
        with rasterio.open(tif) as src:
            # Read only the first band for this label audit.
            arr = src.read(1, masked=True)
            valid = np.asarray(arr.compressed())
            unique, counts = np.unique(valid, return_counts=True)

            # Limit the output size for rasters with many distinct values.
            pairs = sorted(
                zip(unique.tolist(), counts.tolist()),
                key=lambda x: x[1],
                reverse=True
            )
            top_values = ";".join(
                f"{v}:{n}" for v, n in pairs[:20]
            )

            rows.append({
                "filename": tif.name,
                "type": meta.get("type", ""),
                "key": meta.get("key", ""),
                "pre2_date": meta.get("pre2_date", ""),
                "pre1_date": meta.get("pre1_date", ""),
                "post_date": meta.get("post_date", ""),
                "flooding": meta.get("flooding", ""),
                "flooded_tiles": meta.get("flooded_tiles", ""),
                "pflood": meta.get("info", {}).get("pflood", ""),
                "width": src.width,
                "height": src.height,
                "bands": src.count,
                "dtype": src.dtypes[0],
                "nodata": src.nodata,
                "valid_pixel_count": int(valid.size),
                "unique_value_count": int(len(unique)),
                "min_value": float(np.min(valid)) if valid.size else "",
                "max_value": float(np.max(valid)) if valid.size else "",
                "top_values_and_counts": top_values,
                "crs": str(src.crs),
                "transform": str(src.transform),
            })

    except Exception as exc:
        rows.append({
            "filename": tif.name,
            "type": meta.get("type", ""),
            "key": meta.get("key", ""),
            "flooding": meta.get("flooding", ""),
            "error": repr(exc),
        })
        print(f"Raster error: {tif.name}: {exc}")

report = OUT / "step7_gff_label_audit.csv"

if rows:
    fields = sorted({k for row in rows for k in row})
    with report.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

print("Rows audited:", len(rows))
print("Output:", report)
print("Metadata label counts:",
      dict(Counter(str(r.get("flooding", "")) for r in rows)))
print("Raster errors:",
      sum(bool(r.get("error")) for r in rows))
