import csv
import re
from pathlib import Path
from datetime import date
import rasterio

ROOT = Path.cwd()
GFF = ROOT / "GFF"
REPORTS = ROOT / "audit" / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

# Examples:
# 205-1-2017-05-30.tif
# 205-1-2017-05-30-dem-local.tif
# 205-1-2017-05-12-s1.tif
# 205-1_2017-05-30.nc
TIF_RE = re.compile(
    r"^(?P<roi>\d+-\d+)-(?P<date>\d{4}-\d{2}-\d{2})"
    r"(?:-(?P<suffix>.+))?\.tif$",
    re.IGNORECASE,
)
NC_RE = re.compile(
    r"^(?P<roi>\d+-\d+)_(?P<date>\d{4}-\d{2}-\d{2})\.nc$",
    re.IGNORECASE,
)


def scan_tiffs(folder):
    records = []
    for path in (GFF / folder / "rois").glob("*"):
        if not path.is_file():
            continue
        match = TIF_RE.match(path.name)
        if not match:
            continue
        records.append({
            "roi": match.group("roi"),
            "date": match.group("date"),
            "suffix": (match.group("suffix") or "").lower(),
            "path": path,
        })
    return records


def scan_glofas():
    records = []
    folder = GFF / "glofas" / "rois"
    for path in folder.glob("*.nc"):
        match = NC_RE.match(path.name)
        if match:
            records.append({
                "roi": match.group("roi"),
                "date": match.group("date"),
                "path": path,
            })
    return records


def raster_info(path):
    if path is None:
        return {
            "path": "",
            "width": "",
            "height": "",
            "bands": "",
            "crs": "",
            "bounds": "",
            "metadata_status": "MISSING",
        }

    info = {"path": str(path.relative_to(ROOT))}
    try:
        with rasterio.open(path) as src:
            info.update({
                "width": src.width,
                "height": src.height,
                "bands": src.count,
                "crs": str(src.crs),
                "bounds": ",".join(f"{v:.6f}" for v in src.bounds),
                "metadata_status": "OK",
            })
    except Exception as exc:
        info.update({
            "width": "",
            "height": "",
            "bands": "",
            "crs": "",
            "bounds": "",
            "metadata_status": f"ERROR: {exc}",
        })
    return info


base = [
    r for r in scan_tiffs("base")
    if r["suffix"] == ""
]
dem = scan_tiffs("dem")
era5 = scan_tiffs("era5")
s1 = scan_tiffs("s1")
glofas = scan_glofas()

def select(records, roi, target_date, suffix=None):
    return [
        r for r in records
        if r["roi"] == roi
        and r["date"] == target_date
        and (suffix is None or r["suffix"] == suffix)
    ]

rows = []

for target in sorted(base, key=lambda r: (r["roi"], r["date"])):
    roi, target_date = target["roi"], target["date"]

    dem_local = select(dem, roi, target_date, "dem-local")
    dem_context = select(dem, roi, target_date, "dem-context")
    era5_main = select(era5, roi, target_date, "era5")
    era5_land = select(era5, roi, target_date, "era5-land")
    hydro = [
        r for r in glofas
        if r["roi"] == roi and r["date"] == target_date
    ]

    # S1 is often acquired before the target date.
    # Choose the most recent dated S1 observation not after the target.
    prior_s1 = [
        r for r in s1
        if r["roi"] == roi and r["date"] <= target_date
    ]
    prior_s1.sort(key=lambda r: r["date"], reverse=True)
    s1_match = prior_s1[0] if prior_s1 else None

    chosen = {
        "target": target["path"],
        "dem_local": dem_local[0]["path"] if dem_local else None,
        "dem_context": dem_context[0]["path"] if dem_context else None,
        "era5": era5_main[0]["path"] if era5_main else None,
        "era5_land": era5_land[0]["path"] if era5_land else None,
        "glofas": hydro[0]["path"] if hydro else None,
        "s1": s1_match["path"] if s1_match else None,
    }

    row = {
        "roi": roi,
        "target_date": target_date,
        "target_file": target["path"].name,
        "dem_local_found": bool(dem_local),
        "dem_context_found": bool(dem_context),
        "era5_found": bool(era5_main),
        "era5_land_found": bool(era5_land),
        "glofas_found": bool(hydro),
        "prior_s1_found": bool(s1_match),
        "prior_s1_date": s1_match["date"] if s1_match else "",
        "days_between_s1_and_target": (
            (date.fromisoformat(target_date)
             - date.fromisoformat(s1_match["date"])).days
            if s1_match else ""
        ),
    }

    for key, path in chosen.items():
        metadata = raster_info(path)
        for field in ("path", "width", "height", "bands",
                      "crs", "bounds", "metadata_status"):
            row[f"{key}_{field}"] = metadata[field]

    # Exact bounding-box equality is a diagnostic, not a requirement:
    # different products may legitimately use different grids.
    target_bounds = row["target_bounds"]
    for key in ("dem_local", "dem_context", "era5", "era5_land", "s1"):
        row[f"{key}_bounds_equal_target"] = (
            bool(target_bounds)
            and bool(row[f"{key}_bounds"])
            and target_bounds == row[f"{key}_bounds"]
        )

    rows.append(row)

output = REPORTS / "step4_gff_alignment.csv"
if rows:
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

summary = REPORTS / "step4_gff_alignment_summary.txt"
with summary.open("w", encoding="utf-8") as f:
    f.write(f"Base target TIFFs: {len(base)}\n")
    f.write(f"DEM TIFFs: {len(dem)}\n")
    f.write(f"ERA5 TIFFs: {len(era5)}\n")
    f.write(f"S1 TIFFs: {len(s1)}\n")
    f.write(f"GLOFAS NetCDF files: {len(glofas)}\n")
    f.write(f"Target observations audited: {len(rows)}\n\n")

    for field in (
        "dem_local_found", "dem_context_found", "era5_found",
        "era5_land_found", "glofas_found", "prior_s1_found"
    ):
        count = sum(r[field] for r in rows)
        f.write(f"{field}: {count}/{len(rows)}\n")

    for key in ("target", "dem_local", "dem_context",
                "era5", "era5_land", "s1"):
        errors = sum(
            str(r[f"{key}_metadata_status"]).startswith("ERROR")
            for r in rows
        )
        f.write(f"{key} metadata errors: {errors}\n")

print(f"Base target TIFFs audited: {len(base)}")
print(f"Report: {output}")
print(f"Summary: {summary}")
