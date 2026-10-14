import json
from pathlib import Path
import numpy as np
import rasterio

folder = Path("GFF/base/rois")
samples = []

for tif in sorted(folder.glob("*.tif")):
    meta_path = tif.with_name(tif.stem + "-meta.json")
    if not meta_path.exists():
        continue

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        continue

    kind = meta.get("type", "unknown")
    if kind not in ("kurosiwo", "generated"):
        continue

    samples.append((kind, tif, meta))

# Choose up to 3 observations from each type.
chosen = []
for kind in ("kurosiwo", "generated"):
    chosen.extend([x for x in samples if x[0] == kind][:3])

for kind, tif, meta in chosen:
    try:
        with rasterio.open(tif) as src:
            # Read at most 128 x 128 pixels from the first band.
            h = min(128, src.height)
            w = min(128, src.width)
            arr = src.read(1, window=((0, h), (0, w)), masked=True)
            values = arr.compressed()
            unique = np.unique(values)

            print(f"\nFile: {tif.name}")
            print(f"Type: {kind}")
            print(f"Flooding metadata: {meta.get('flooding')}")
            print(f"Flooded tiles: {meta.get('flooded_tiles', 'N/A')}")
            print(f"Raster: {src.width}x{src.height}, bands={src.count}, dtype={src.dtypes[0]}")
            print(f"CRS present: {src.crs is not None}")
            print(f"Sample valid pixels: {values.size}")
            print(f"Sample min/max: {values.min() if values.size else 'N/A'} / {values.max() if values.size else 'N/A'}")
            print(f"Unique values in sample: {unique[:15].tolist()}")
            print(f"Unique count in sample: {len(unique)}")
    except Exception as exc:
        print(f"\nERROR: {tif.name}: {exc}")

print(f"\nFinished. Sampled {len(chosen)} rasters.")
