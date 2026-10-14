from pathlib import Path
from collections import Counter, defaultdict
import csv
import json

# Project root directory
ROOT = Path(r"C:\Disaster Progression Dataset")
OUT = ROOT / "audit" / "reports"
OUT.mkdir(parents=True, exist_ok=True)

# Dataset locations
DATASETS = {
    "GFF": ROOT / "GFF",
    "SEN12FLOOD": ROOT / "SEN12FLOOD",
    "TS-SatFire": ROOT / "TS-SatFire-selected",
}

records = []
summary = defaultdict(Counter)

print("Starting dataset inventory...")

for dataset, base in DATASETS.items():
    if not base.exists():
        print(f"[MISSING DATASET] {base}")
        continue

    print(f"\nScanning {dataset}...")

    for path in base.rglob("*"):
        if not path.is_file():
            continue

        try:
            size = path.stat().st_size
        except OSError as exc:
            print(f"[WARNING] Cannot access {path}: {exc}")
            size = -1

        rel = path.relative_to(base)
        parts = rel.parts
        ext = path.suffix.lower() or "[no extension]"
        top_folder = parts[0] if len(parts) > 1 else "[root]"

        record = {
            "dataset": dataset,
            "relative_path": str(rel),
            "filename": path.name,
            "extension": ext,
            "parent_folder": str(rel.parent),
            "top_folder": top_folder,
            "size_bytes": size,
            "size_mb": round(size / (1024 * 1024), 3)
            if size >= 0 else "",
        }
        records.append(record)

        summary[dataset]["files"] += 1
        summary[dataset][f"ext:{ext}"] += 1
        summary[dataset][f"top:{top_folder}"] += 1

        if size >= 0:
            summary[dataset]["bytes"] += size

# Save complete file inventory
inventory_path = OUT / "step1_file_inventory.csv"

fields = [
    "dataset",
    "relative_path",
    "filename",
    "extension",
    "parent_folder",
    "top_folder",
    "size_bytes",
    "size_mb",
]

with inventory_path.open(
    "w", newline="", encoding="utf-8-sig"
) as file:
    writer = csv.DictWriter(file, fieldnames=fields)
    writer.writeheader()
    writer.writerows(records)

# Build dataset summary
summary_json = {}

for dataset, counts in summary.items():
    summary_json[dataset] = {
        "files": counts["files"],
        "total_size_gb": round(
            counts["bytes"] / (1024 ** 3), 3
        ),
        "extensions": {
            key[4:]: value
            for key, value in counts.items()
            if key.startswith("ext:")
        },
        "top_level_folders": {
            key[4:]: value
            for key, value in counts.items()
            if key.startswith("top:")
        },
    }

summary_path = OUT / "step1_inventory_summary.json"
summary_path.write_text(
    json.dumps(summary_json, indent=2),
    encoding="utf-8",
)

# Print summary
print("\n" + "=" * 55)
print("DATASET INVENTORY SUMMARY")
print("=" * 55)

for dataset, info in summary_json.items():
    print(f"\nDataset: {dataset}")
    print(f"Total files: {info['files']}")
    print(f"Total size: {info['total_size_gb']} GB")
    print("File extensions:")

    for ext, count in sorted(info["extensions"].items()):
        print(f"  {ext}: {count}")

    print("Top-level folders:")

    for folder, count in sorted(
        info["top_level_folders"].items()
    ):
        print(f"  {folder}: {count}")

print("\nInventory completed.")
print(f"CSV report: {inventory_path}")
print(f"JSON summary: {summary_path}")
