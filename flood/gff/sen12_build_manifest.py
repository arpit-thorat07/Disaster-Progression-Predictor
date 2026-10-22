
import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "SEN12FLOOD"
OUT = ROOT / "data" / "sen12_manifest.csv"

OUT.parent.mkdir(parents=True, exist_ok=True)

with (DATA / "S1list.json").open(
    encoding="utf-8"
) as f:
    metadata = json.load(f)

rows = []
missing = []
labels = Counter()
folders = set()

for folder_id, group in metadata.items():

    folder = DATA / folder_id

    if not folder.is_dir():
        missing.append((folder_id, "folder missing"))
        continue

    for key, entry in group.items():

        if not isinstance(entry, dict):
            continue

        if "filename" not in entry:
            continue

        if "FLOODING" not in entry:
            continue

        base = entry["filename"]

        vv = folder / f"{base}_corrected_VV.tif"
        vh = folder / f"{base}_corrected_VH.tif"

        # Check exact filenames rather than
        # selecting arbitrary files by date.
        if not vv.exists() or not vh.exists():
            missing.append((folder_id, base))
            continue

        flooding = entry["FLOODING"]

        if not isinstance(flooding, bool):
            raise ValueError(
                f"Non-Boolean flood label: {entry}"
            )

        label = int(flooding)

        rows.append({
            "folder": folder_id,
            "date": entry["date"],
            "orbit": entry.get("orbit", ""),
            "flooding": label,
            "vv_path": str(vv.relative_to(ROOT)),
            "vh_path": str(vh.relative_to(ROOT)),
        })

        labels[label] += 1
        folders.add(folder_id)

# Identify accidental duplicates before training.
seen = set()

for row in rows:
    identifier = (
        row["folder"],
        row["vv_path"],
        row["vh_path"]
    )

    if identifier in seen:
        raise RuntimeError(
            f"Duplicate acquisition: {identifier}"
        )

    seen.add(identifier)

rows.sort(
    key=lambda r: (
        r["folder"],
        r["date"],
        r["vv_path"]
    )
)

if not rows:
    raise RuntimeError(
        "No valid SEN12 Sentinel-1 pairs found"
    )

with OUT.open(
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "folder",
            "date",
            "orbit",
            "flooding",
            "vv_path",
            "vh_path",
        ]
    )

    writer.writeheader()
    writer.writerows(rows)

print("SEN12-FLOOD TRAINING MANIFEST")

print("Valid VV/VH acquisitions:", len(rows))
print("Folders represented:", len(folders))
print("Flood-positive acquisitions:", labels[1])
print("Flood-negative acquisitions:", labels[0])
print("Missing or incomplete pairs:", len(missing))

print("\nFIRST FIVE EXAMPLES")

for row in rows[:5]:
    print(
        row["folder"],
        row["date"],
        "FLOOD" if row["flooding"] else "NO FLOOD",
        row["orbit"]
    )

print("\nFIRST TEN MISSING PAIRS")

for item in missing[:10]:
    print(item)

print("\nSaved:", OUT)
print("SEN12 MANIFEST COMPLETED")
