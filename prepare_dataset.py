
import csv
import hashlib
import random
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = PROJECT_ROOT / "archive"
OUTPUT_ROOT = PROJECT_ROOT / "dataset_brain_tumor"
TRAIN_CSV = OUTPUT_ROOT / "train.csv"
VAL_CSV = OUTPUT_ROOT / "val.csv"
TEST_CSV = OUTPUT_ROOT / "test.csv"

VAL_RATIO = 0.2
SEED = 42
CLASSES = ("glioma_tumor", "meningioma_tumor", "pituitary_tumor", "no_tumor")
SOURCE_TO_TARGET = {
    "glioma_tumor": (1, "tumor"),
    "meningioma_tumor": (1, "tumor"),
    "pituitary_tumor": (1, "tumor"),
    "no_tumor": (0, "not_tumor"),
}
TARGET_CLASSES = ("not_tumor", "tumor")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_images(folder: Path):
    return sorted(
        path for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def collect_partition(partition: str, reserved_hashes=None):
    reserved_hashes = reserved_hashes or {}
    unique_by_hash = {}
    duplicates_removed = 0
    reserved_removed = 0

    for class_name in CLASSES:
        class_dir = SOURCE_ROOT / partition / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class directory: {class_dir}")
        images = list_images(class_dir)
        if not images:
            raise ValueError(f"No supported images found in: {class_dir}")

        label, target_name = SOURCE_TO_TARGET[class_name]
        for image_path in images:
            digest = hashlib.sha256(image_path.read_bytes()).digest()
            reserved = reserved_hashes.get(digest)
            if reserved is not None:
                if reserved[1] != label:
                    raise ValueError(
                        "Identical image content has conflicting labels: "
                        f"{reserved[0]} and {image_path}"
                    )
                reserved_removed += 1
                continue
            previous = unique_by_hash.get(digest)
            if previous is not None:
                if previous[1] != label:
                    raise ValueError(
                        "Identical image content has conflicting labels: "
                        f"{previous[0]} and {image_path}"
                    )
                duplicates_removed += 1
                continue
            unique_by_hash[digest] = (image_path, label, target_name)

    return unique_by_hash, duplicates_removed, reserved_removed


def stratified_split(records):
    rng = random.Random(SEED)
    train_rows, val_rows = [], []

    for class_name in TARGET_CLASSES:
        label = TARGET_CLASSES.index(class_name)
        class_records = sorted(
            (record for record in records if record[1] == label),
            key=lambda record: str(record[0]),
        )
        if len(class_records) < 2:
            raise ValueError(f"At least two unique images are required for {class_name}")
        rng.shuffle(class_records)
        validation_count = min(
            len(class_records) - 1,
            max(1, round(len(class_records) * VAL_RATIO)),
        )
        val_rows.extend(class_records[:validation_count])
        train_rows.extend(class_records[validation_count:])

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    return train_rows, val_rows


def write_manifest(records, destination: Path):
    temporary = destination.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("image", "label", "class_name"))
        for image_path, label, class_name in records:
            writer.writerow((image_path.relative_to(PROJECT_ROOT), label, class_name))
    temporary.replace(destination)


def main():
    test_by_hash, test_duplicates, _ = collect_partition("Testing")
    training_by_hash, training_duplicates, cross_partition_removed = collect_partition(
        "Training", reserved_hashes=test_by_hash
    )
    train_rows, val_rows = stratified_split(list(training_by_hash.values()))
    test_rows = list(test_by_hash.values())
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    write_manifest(train_rows, TRAIN_CSV)
    write_manifest(val_rows, VAL_CSV)
    write_manifest(test_rows, TEST_CSV)

    train_counts = Counter(row[2] for row in train_rows)
    val_counts = Counter(row[2] for row in val_rows)
    test_counts = Counter(row[2] for row in test_rows)
    print(
        "Duplicates removed: "
        f"training={training_duplicates}, test={test_duplicates}, "
        f"training images also present in test={cross_partition_removed}"
    )
    print(f"Training: {len(train_rows)} -> {TRAIN_CSV.relative_to(PROJECT_ROOT)}")
    print(f"Validation: {len(val_rows)} -> {VAL_CSV.relative_to(PROJECT_ROOT)}")
    print(f"Test: {len(test_rows)} -> {TEST_CSV.relative_to(PROJECT_ROOT)}")
    for class_name in TARGET_CLASSES:
        print(
            f"  {class_name}: train={train_counts[class_name]}, "
            f"val={val_counts[class_name]}, test={test_counts[class_name]}"
        )


if __name__ == "__main__":
    main()
