from pathlib import Path
import shutil
import random
import csv


DATASET_ROOT = Path("Dataset")
TRAIN_DIR_NAME = "Training"
TEST_DIR_NAME  = "Testing"

OUT_ROOT = Path("dataset_brain_tumor")
OUT_TRAIN_DIR = OUT_ROOT / "train_images"
OUT_VAL_DIR   = OUT_ROOT / "val_images"
OUT_TEST_DIR  = OUT_ROOT / "test_images"

TRAIN_CSV = OUT_ROOT / "train.csv"
VAL_CSV   = OUT_ROOT / "val.csv"
TEST_CSV  = OUT_ROOT / "test.csv"

VAL_RATIO = 0.2
SEED = 42
COPY_FILES = True  

CLASSES = [
    "glioma_tumor",      
    "meningioma_tumor",
    "pituitary_tumor",
    "no_tumor"
]

CLASS_TO_LABEL = {cls: i for i, cls in enumerate(CLASSES)}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def list_images(folder: Path):
    return [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]


def unique_name(target_dir: Path, filename: str) -> str:

    base = Path(filename).stem
    ext = Path(filename).suffix
    candidate = filename
    k = 1
    while (target_dir / candidate).exists():
        candidate = f"{base}_{k}{ext}"
        k += 1
    return candidate


def copy_or_move(src: Path, dst: Path, copy_files: bool):
    if copy_files:
        shutil.copy2(src, dst)
    else:
        shutil.move(str(src), str(dst))


def write_csv(rows, csv_path: Path):
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "label"])
        w.writerows(rows)


def main():
    random.seed(SEED)

    train_root = DATASET_ROOT / TRAIN_DIR_NAME
    test_root  = DATASET_ROOT / TEST_DIR_NAME

    safe_mkdir(OUT_TRAIN_DIR)
    safe_mkdir(OUT_VAL_DIR)
    safe_mkdir(OUT_TEST_DIR)
    safe_mkdir(OUT_ROOT)


    train_rows = []
    val_rows = []

    for cls in CLASSES:
        cls_dir = train_root / cls
        if not cls_dir.exists():
            raise FileNotFoundError(f"No folder found: {cls_dir}")

        imgs = list_images(cls_dir)
        if len(imgs) == 0:
            raise ValueError(f"img not found: {cls_dir}")

        random.shuffle(imgs)
        n_val = int(round(len(imgs) * VAL_RATIO))
        val_imgs = imgs[:n_val]
        train_imgs = imgs[n_val:]

        label = CLASS_TO_LABEL[cls]

        # train split
        for src in train_imgs:
            new_name = f"{src.stem}_{cls}{src.suffix.lower()}"
            new_name = unique_name(OUT_TRAIN_DIR, new_name)
            dst = OUT_TRAIN_DIR / new_name
            copy_or_move(src, dst, COPY_FILES)
            train_rows.append([new_name, label])

        # val split
        for src in val_imgs:
            new_name = f"{src.stem}_{cls}{src.suffix.lower()}"
            new_name = unique_name(OUT_VAL_DIR, new_name)
            dst = OUT_VAL_DIR / new_name
            copy_or_move(src, dst, COPY_FILES)
            val_rows.append([new_name, label])

    write_csv(train_rows, TRAIN_CSV)
    write_csv(val_rows, VAL_CSV)


    test_rows = []

    for cls in CLASSES:
        cls_dir = test_root / cls
        if not cls_dir.exists():
            raise FileNotFoundError(f"No folder found: {cls_dir}")

        imgs = list_images(cls_dir)
        label = CLASS_TO_LABEL[cls]

        for src in imgs:
            new_name = f"test_{src.stem}_{cls}{src.suffix.lower()}"
            new_name = unique_name(OUT_TEST_DIR, new_name)
            dst = OUT_TEST_DIR / new_name
            copy_or_move(src, dst, COPY_FILES)
            test_rows.append([new_name, label])

    write_csv(test_rows, TEST_CSV)


    print(f"- Train images: {OUT_TRAIN_DIR} | CSV: {TRAIN_CSV}")
    print(f"- Val images:   {OUT_VAL_DIR}   | CSV: {VAL_CSV}")
    print(f"- Test images:  {OUT_TEST_DIR}  | CSV: {TEST_CSV}")
    print("Label mapping:", CLASS_TO_LABEL)


if __name__ == "__main__":
    main()
