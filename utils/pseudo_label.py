"""
Pseudo-label generation for skin lesion classification dataset.
Converts class-labeled images → YOLO-format bounding boxes via CV segmentation.

Strategy:
  1. Resize image to target size first (fast)
  2. LAB color space + Otsu thresholding on L and A channels
  3. Morphological cleanup
  4. Largest contour → bounding rect → normalized YOLO format
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import random
import argparse

# --- Class mapping: original folder name → detection class ID ---
CLASS_MAP = {
    "SkinCancer":          0,   # skin_cancer
    "Moles":               1,   # nevus
    "Actinic_Keratosis":   2,   # actinic_keratosis
    "Seborrh_Keratoses":   3,   # seborrheic_keratosis
    "Benign_tumors":       4,   # benign_tumor
    "Vascular_Tumors":     5,   # vascular_lesion
    "Warts":               6,   # wart
    "Infestations_Bites":  7,   # infestation_bite
}

CLASS_FRIENDLY = [
    "skin_cancer",
    "nevus",
    "actinic_keratosis",
    "seborrheic_keratosis",
    "benign_tumor",
    "vascular_lesion",
    "wart",
    "infestation_bite",
]


def segment_lesion(image_bgr: np.ndarray) -> tuple:
    """
    Segment the skin lesion from background using LAB + Otsu thresholding.
    Runs on already-resized image (fast).

    Returns (mask, bbox_xywh_norm) or (None, None) on failure.
    """
    h, w = image_bgr.shape[:2]

    # Convert to LAB — L channel has good skin/lesion contrast
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    # Otsu on L-channel (dark regions = potential lesion)
    _, thresh_l = cv2.threshold(L, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Otsu on A-channel (redness = potential lesion)
    _, thresh_a = cv2.threshold(A, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Combine: union of L-dark and A-red regions
    combined = cv2.bitwise_or(thresh_l, thresh_a)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    # Use largest contour
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)

    # If too small (<3% of image), use near-full-image bbox as fallback
    if area < 0.03 * h * w:
        return cleaned, None

    x, y, bw, bh = cv2.boundingRect(largest)

    # If bbox covers >85% of image, likely bad — use near-full-image fallback
    if (bw * bh) > 0.85 * h * w:
        return cleaned, None

    # Add small padding (5% of box size)
    px = int(bw * 0.05)
    py = int(bh * 0.05)
    x = max(0, x - px)
    y = max(0, y - py)
    bw = min(w - x, bw + 2 * px)
    bh = min(h - y, bh + 2 * py)

    # Normalize to YOLO format: cx cy w h (all 0-1)
    cx = (x + bw / 2) / w
    cy = (y + bh / 2) / h
    nw = bw / w
    nh = bh / h

    return cleaned, (cx, cy, nw, nh)


def process_image(img_path: Path, class_id: int, out_img_dir: Path, out_lbl_dir: Path, img_size: int = 640):
    """
    Process a single image: resize first, then segment on the small image, save image + label.
    Returns True on success.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return False

    # Resize FIRST — segmentation runs on the small image for speed
    img_resized = cv2.resize(img, (img_size, img_size))

    # Segment on the resized image
    _, bbox = segment_lesion(img_resized)

    if bbox is None:
        # Fallback: near-full-image bbox
        bbox = (0.5, 0.5, 0.9, 0.9)

    # Save image
    out_name = img_path.stem + ".jpg"
    cv2.imwrite(str(out_img_dir / out_name), img_resized, [cv2.IMWRITE_JPEG_QUALITY, 95])

    # Save label
    cx, cy, nw, nh = bbox
    label_path = out_lbl_dir / (img_path.stem + ".txt")
    label_path.write_text(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

    return True


def build_dataset(
    raw_root: str,
    out_root: str,
    img_size: int = 640,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
):
    """
    Main pipeline: read raw classification dataset, generate YOLO-format data.
    """
    raw_root = Path(raw_root)
    out_root = Path(out_root)

    # Collect all images from train+test dirs
    all_samples = []  # (img_path, class_id, original_split)
    for split in ["Train", "Test"]:
        split_dir = raw_root / split
        if not split_dir.exists():
            continue
        for class_name, class_id in CLASS_MAP.items():
            class_dir = split_dir / class_name
            if not class_dir.exists():
                print(f"  [WARN] Missing class dir: {class_dir}")
                continue
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                    all_samples.append((img_path, class_id, split))

    print(f"Found {len(all_samples)} images across {len(CLASS_MAP)} classes")

    # Shuffle and split
    random.seed(42)
    random.shuffle(all_samples)

    n = len(all_samples)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    splits = {
        "train": all_samples[:n_train],
        "val": all_samples[n_train:n_train + n_val],
        "test": all_samples[n_train + n_val:],
    }

    # Create output dirs
    for s in ["train", "val", "test"]:
        (out_root / "images" / s).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / s).mkdir(parents=True, exist_ok=True)

    # Process each split
    stats = {}
    for split_name, samples in splits.items():
        print(f"\nProcessing {split_name} ({len(samples)} images)...")
        img_dir = out_root / "images" / split_name
        lbl_dir = out_root / "labels" / split_name
        class_counts = {}

        for img_path, class_id, _ in tqdm(samples, desc=split_name):
            success = process_image(img_path, class_id, img_dir, lbl_dir, img_size)
            if success:
                class_counts[class_id] = class_counts.get(class_id, 0) + 1

        stats[split_name] = class_counts
        print(f"  {split_name}: {class_counts}")

    # Write dataset.yaml
    yaml_path = out_root / "dataset.yaml"
    yaml_content = f"""# Skin Lesion Detection Dataset (YOLO format)
# Generated via pseudo-labeling from Skin Disease classification dataset

path: {out_root.absolute()}
train: images/train
val: images/val
test: images/test

nc: {len(CLASS_MAP)}
names: {CLASS_FRIENDLY}
"""
    yaml_path.write_text(yaml_content)
    print(f"\nDataset YAML written to: {yaml_path}")

    # Print summary
    print("\n" + "=" * 50)
    print("DATASET BUILD COMPLETE")
    print("=" * 50)
    for s in ["train", "val", "test"]:
        total = sum(stats.get(s, {}).values())
        print(f"  {s}: {total} images")
        for cid, count in sorted(stats.get(s, {}).items()):
            print(f"    {CLASS_FRIENDLY[cid]}: {count}")

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate pseudo-labels for skin lesion dataset")
    parser.add_argument("--raw", type=str, required=True, help="Path to raw dataset root")
    parser.add_argument("--out", type=str, required=True, help="Path to output YOLO dataset root")
    parser.add_argument("--img-size", type=int, default=640, help="Target image size (default: 640)")
    args = parser.parse_args()

    build_dataset(args.raw, args.out, args.img_size)
