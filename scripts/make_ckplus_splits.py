from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split


# Supported image file extensions for dataset processing
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def sha1_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """
    Calculate SHA1 hash of a file for integrity verification.
    
    Args:
        path: Path to the file to hash
        chunk_size: Size of chunks to read (default: 1MB)
        
    Returns:
        SHA1 hash as hexadecimal string
    """
    h = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def auto_detect_raw_dir(data_raw_root: Path) -> Path:
    """
    If user doesn't pass --raw_dir, pick a reasonable default:
    - If there's only one directory under data/raw, use it.
    - Else prefer one containing "ck" or "ckplus" in its name.
    """
    candidates = [p for p in data_raw_root.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No dataset folder found under: {data_raw_root}")

    if len(candidates) == 1:
        return candidates[0]

    # Prefer likely CK+ folders
    preferred = []
    for p in candidates:
        name = p.name.lower()
        if "ck" in name or "ckplus" in name or "cohn" in name:
            preferred.append(p)

    if len(preferred) == 1:
        return preferred[0]

    # Fall back: first directory (stable alphabetical)
    candidates = sorted(candidates, key=lambda x: x.name.lower())
    return candidates[0]


def find_class_folders(raw_dir: Path) -> list[Path]:
    """
    Find class folders in the raw dataset directory.
    
    CK+ datasets typically organize images by emotion categories
    (e.g., anger, happy, sad, etc.) in separate subfolders.
    
    Args:
        raw_dir: Directory containing class subfolders
        
    Returns:
        Sorted list of class directory paths
        
    Raises:
        FileNotFoundError: If no class folders are found
    """
    # Only take direct subfolders as classes (common Kaggle layout)
    class_dirs = [p for p in raw_dir.iterdir() if p.is_dir()]
    if not class_dirs:
        raise FileNotFoundError(
            f"No class folders found inside: {raw_dir}\n"
            f"Expected something like: {raw_dir}/anger/*.png, {raw_dir}/happy/*.png, ..."
        )
    return sorted(class_dirs, key=lambda p: p.name.lower())


def collect_images(class_dir: Path) -> list[Path]:
    """
    Recursively collect all image files from a class directory.
    
    Args:
        class_dir: Directory containing images for a specific class
        
    Returns:
        Sorted list of image file paths
    """
    paths: list[Path] = []
    for p in class_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            paths.append(p)
    return sorted(paths)


def main():
    """
    Main function to process CK+ dataset and create train/val/test splits.
    
    This script:
    1. Detects the CK+ dataset directory structure
    2. Collects all images from class folders
    3. Optionally verifies image integrity and extracts metadata
    4. Creates stratified train/validation/test splits
    5. Saves metadata and split files for ML pipeline
    """
    parser = argparse.ArgumentParser(description="Create manifest + stratified splits for CK+ Kaggle folder dataset.")
    parser.add_argument("--raw_root", type=str, default="data/raw", help="Root folder containing raw datasets (default: data/raw).")
    parser.add_argument("--raw_dir", type=str, default="", help="Exact dataset folder under raw_root (e.g. ckplus_kaggle). If empty, auto-detect.")
    parser.add_argument("--out_dir", type=str, default="data/processed/ckplus", help="Output folder (default: data/processed/ckplus).")
    parser.add_argument("--train_size", type=float, default=0.70, help="Train split fraction (default: 0.70).")
    parser.add_argument("--val_size", type=float, default=0.15, help="Val split fraction (default: 0.15).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument("--verify", action="store_true", help="Open each image and record width/height/mode + sha1 (slower but safer).")
    args = parser.parse_args()

    # Setup input and output directories
    raw_root = Path(args.raw_root)
    raw_root.mkdir(parents=True, exist_ok=True)

    # Determine the raw dataset directory (either specified or auto-detected)
    if args.raw_dir.strip():
        raw_dir = raw_root / args.raw_dir.strip()
        if not raw_dir.exists():
            raise FileNotFoundError(f"--raw_dir not found: {raw_dir}")
    else:
        raw_dir = auto_detect_raw_dir(raw_root)

    # Create output directories
    out_dir = Path(args.out_dir)
    splits_dir = out_dir / "splits"
    out_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    # Discover class structure and create label mapping
    class_dirs = find_class_folders(raw_dir)
    class_names = [p.name for p in class_dirs]
    class_to_id = {name: i for i, name in enumerate(class_names)}  # stable alphabetical mapping

    # Collect metadata for all images
    rows = []
    broken = 0

    # Process each class folder and its images
    for cdir in class_dirs:
        cname = cdir.name
        label = class_to_id[cname]
        imgs = collect_images(cdir)

        # Create metadata record for each image
        for img_path in imgs:
            record = {
                "path": str(img_path.as_posix()),
                "label": int(label),
                "label_name": cname,
            }

            # Optional: verify image integrity and extract metadata
            if args.verify:
                try:
                    with Image.open(img_path) as im:
                        record["width"], record["height"] = im.size
                        record["mode"] = im.mode
                    record["sha1"] = sha1_file(img_path)
                except Exception:
                    broken += 1
                    continue

            rows.append(record)

    # Validate that we found images and create DataFrame
    if not rows:
        raise RuntimeError(f"No images found in {raw_dir}. Check folder layout and extensions.")

    # Create and shuffle the metadata DataFrame
    df = pd.DataFrame(rows)
    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    # Save dataset metadata and class mapping
    df.to_csv(out_dir / "metadata.csv", index=False)
    with (out_dir / "class_map.json").open("w", encoding="utf-8") as f:
        json.dump({"classes": class_names, "class_to_id": class_to_id}, f, indent=2)

    # Create stratified train/validation/test splits
    train_size = args.train_size
    val_size = args.val_size
    test_size = 1.0 - train_size - val_size
    if test_size <= 0:
        raise ValueError("train_size + val_size must be < 1.0")

    # First split: train vs (val + test)
    train_df, temp_df = train_test_split(
        df,
        test_size=(1.0 - train_size),
        random_state=args.seed,
        stratify=df["label"],
    )

    # Second split: val vs test from remaining data
    val_frac_of_temp = val_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - val_frac_of_temp),
        random_state=args.seed,
        stratify=temp_df["label"],
    )

    # Save split files
    train_df.to_csv(splits_dir / "train.csv", index=False)
    val_df.to_csv(splits_dir / "val.csv", index=False)
    test_df.to_csv(splits_dir / "test.csv", index=False)

    # Print summary statistics
    print("\n=== CK+ Split Builder ===")
    print(f"Raw dataset folder: {raw_dir}")
    print(f"Classes ({len(class_names)}): {class_names}")
    print(f"Total samples: {len(df)} (broken skipped: {broken})\n")

    print("Class counts:")
    print(df["label_name"].value_counts().to_string())
    print("\nSplit sizes:")
    print(f"  train: {len(train_df)}")
    print(f"  val:   {len(val_df)}")
    print(f"  test:  {len(test_df)}")

    print("\nWrote:")
    print(f"  {out_dir / 'metadata.csv'}")
    print(f"  {out_dir / 'class_map.json'}")
    print(f"  {splits_dir / 'train.csv'}")
    print(f"  {splits_dir / 'val.csv'}")
    print(f"  {splits_dir / 'test.csv'}")


# Script entry point
if __name__ == "__main__":
    main()
