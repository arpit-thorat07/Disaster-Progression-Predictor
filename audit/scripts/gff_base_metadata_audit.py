import json
import csv
import re
from pathlib import Path
from collections import Counter

root = Path("GFF/base/rois")
out = Path("audit/reports")
out.mkdir(parents=True, exist_ok=True)

rows = []
errors = []

for tif in sorted(root.glob("*.tif")):
    name = tif.name
    meta_path = tif.with_name(tif.stem + "-meta.json")

    # Extract all dates from the filename.
    dates = re.findall(r"\d{4}-\d{2}-\d{2}", name)

    row = {
        "filename": name,
        "file_size_bytes": tif.stat().st_size,
        "filename_dates": ";".join(dates),
        "metadata_file": meta_path.name if meta_path.exists() else "",
        "metadata_status": "",
        "type": "",
        "key": "",
        "pre1_date": "",
        "post_date": "",
        "flood_date": "",
        "flooding": "",
        "pflood": "",
        "floodmap": "",
        "gvalid": "",
    }

    if not meta_path.exists():
        row["metadata_status"] = "MISSING"
    else:
        try:
            with meta_path.open(encoding="utf-8") as f:
                data = json.load(f)

            row.update({
                "metadata_status": "OK",
                "type": data.get("type", ""),
                "key": data.get("key", ""),
                "pre1_date": data.get("pre1_date", ""),
                "post_date": data.get("post_date", ""),
                "flooding": data.get("flooding", ""),
                "floodmap": data.get("floodmap", ""),
                "flood_date": data.get("info", {}).get("flood_date", ""),
                "pflood": data.get("info", {}).get("pflood", ""),
                "gvalid": data.get("info", {}).get("gvalid", ""),
            })
        except Exception as exc:
            row["metadata_status"] = "ERROR: " + str(exc)
            errors.append((name, str(exc)))

    rows.append(row)

report = out / "step6_gff_base_metadata.csv"
with report.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print("Base TIFFs:", len(rows))
print("Metadata files found:", sum(r["metadata_status"] == "OK" for r in rows))
print("Metadata missing:", sum(r["metadata_status"] == "MISSING" for r in rows))
print("Metadata errors:", len(errors))
print("Filename date-count distribution:",
      dict(Counter(len(r["filename_dates"].split(";"))
                   if r["filename_dates"] else 0 for r in rows)))
print("Metadata type distribution:",
      dict(Counter(r["type"] or "(blank)" for r in rows)))
print("Flooding field distribution:",
      dict(Counter(str(r["flooding"]) for r in rows)))
print("Report:", report)

print("\nFirst 10 records:")
for r in rows[:10]:
    print(
        r["filename"],
        "| key:", r["key"],
        "| pre1:", r["pre1_date"],
        "| post:", r["post_date"],
        "| flood_date:", r["flood_date"],
        "| metadata:", r["metadata_status"]
    )
