
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "data" / "sen12_manifest.csv"
OUTPUT = ROOT / "data" / "sen12_splits.csv"

SEED = 42

if OUTPUT.exists():
    raise FileExistsError(
        f"Split file already exists: {OUTPUT}"
    )

with MANIFEST.open(
    newline="", encoding="utf-8"
) as f:
    rows = list(csv.DictReader(f))

groups = defaultdict(list)

for row in rows:
    groups[row["folder"]].append(row)

folders = sorted(groups)
rng = random.Random(SEED)
rng.shuffle(folders)

#Split entire location folders, never individual images
n = len(folders)

n_train = round(n * 0.70)
n_val = round(n * 0.15)

train_folders = set(folders[:n_train])
val_folders = set(
    folders[n_train:n_train + n_val]
)
test_folders = set(
    folders[n_train + n_val:]
)

assert not (train_folders & val_folders)
assert not (train_folders & test_folders)
assert not (val_folders & test_folders)

for row in rows:
    folder = row["folder"]

    if folder in train_folders:
        row["split"] = "train"
    elif folder in val_folders:
        row["split"] = "val"
    else:
        row["split"] = "test"

summary = defaultdict(Counter)

for row in rows:
    split = row["split"]
    label = int(row["flooding"])

    summary[split]["total"] += 1
    summary[split][f"label_{label}"] += 1

#All splits must contain both classes
for split in ("train", "val", "test"):
    if (
        summary[split]["label_0"] == 0
        or summary[split]["label_1"] == 0
    ):
        raise RuntimeError(
            f"Split {split} lacks one class. "
            "Choose another seed before saving."
        )

with OUTPUT.open(
    "w", newline="", encoding="utf-8"
) as f:
    writer = csv.DictWriter(
        f, fieldnames=list(rows[0].keys())
    )
    writer.writeheader()
    writer.writerows(rows)

print("SEN12-FLOOD GROUPED SPLIT")

print("Seed:", SEED)
print("Total acquisitions:", len(rows))
print("Location folders:", len(folders))

for split, group in [
    ("train", train_folders),
    ("val", val_folders),
    ("test", test_folders),
]:
    stats = summary[split]

    positive_rate = (
        stats["label_1"] / stats["total"]
    )

    print(f"\n{split.upper()}")
    print("  Folders:", len(group))
    print("  Acquisitions:", stats["total"])
    print("  Flood:", stats["label_1"])
    print("  No flood:", stats["label_0"])
    print(
        "  Flood fraction:",
        f"{positive_rate:.2%}"
    )

print("\nFolder overlap: NONE")
print("Saved:", OUTPUT)
print("SEN12 SPLIT COMPLETED")