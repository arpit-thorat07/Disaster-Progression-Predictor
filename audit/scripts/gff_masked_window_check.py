from pathlib import Path
import numpy as np
import rasterio

folder = Path("GFF/base/rois")

names = [
    "130-6-2015-09-13.tif",
    "4207-1080030220-2014-11-10-2014-11-22-2014-12-04.tif",
    "4210-2080000350-2014-11-16-2014-11-21-2014-11-28.tif",
    "4210-2080000350-2014-11-21-2014-11-28-2014-12-03.tif",
]

for name in names:
    path = folder / name
    print(f"\n--- {name} ---")

    try:
        with rasterio.open(path) as src:
            print(f"Dimensions: {src.width} x {src.height}")
            print(f"NoData value: {src.nodata}")
            print(f"Mask flags: {src.mask_flag_enums[0]}")

            positions = {
                "top-left": (0, 0),
                "center": (src.height // 2, src.width // 2),
                "bottom-right": (max(0, src.height - 128),
                                 max(0, src.width - 128)),
            }

            for label, (row, col) in positions.items():
                h = min(128, src.height - row)
                w = min(128, src.width - col)
                arr = src.read(
                    1,
                    window=((row, row + h), (col, col + w)),
                    masked=True,
                )
                values = arr.compressed()

                if values.size:
                    print(
                        f"{label}: valid={values.size}, "
                        f"min={values.min()}, max={values.max()}, "
                        f"unique_sample={np.unique(values)[:10].tolist()}"
                    )
                else:
                    print(f"{label}: zero valid pixels")
    except Exception as exc:
        print(f"ERROR: {exc}")

