"""
Pseudo-label generation for skin lesion classification dataset.
Converts class-labeled images → YOLO-format bounding boxes via CV segmentation.

Strategy:
  1. LAB color space conversion
  2. Otsu thresholding on L-channel
  3. Morphological cleanup
  4. Largest contour → bounding rect
  5. Normalize to YOLO format
"""

import cv2
import numpy as np
from pathlib import Path
import shutil
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

CLASS_NAMES = {v: k for k, v in CLASS_MAP.items()}

# Friendly names for dataset.yaml
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


def _mask_to_bbox(mask: np.ndarray, h: int, w: int, min_area_ratio: float = 0.03):
    """
    Convert binary mask to normalized YOLO bbox (cx, cy, nw, nh).
    Returns None if resulting bbox is invalid or too small.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)

    if area < min_area_ratio * h * w:
        return None

    x, y, bw, bh = cv2.boundingRect(largest)

    # Allow full-image bboxes only if the mask truly covers the image
    bbox_area_ratio = (bw * bh) / (h * w)
    if bbox_area_ratio > 0.85:
        # Too large — likely bad segmentation, reject
        return None

    # Add padding (3% of box size)
    px = int(bw * 0.03)
    py = int(bh * 0.03)
    x = max(0, x - px)
    y = max(0, y - py)
    bw = min(w - x, bw + 2 * px)
    bh = min(h - y, bh + 2 * py)

    cx = (x + bw / 2) / w
    cy = (y + bh / 2) / h
    nw = bw / w
    nh = bh / h

    return (cx, cy, nw, nh)


def _strategy_grabcut(image_bgr: np.ndarray) -> np.ndarray:
    """
    GrabCut segmentation: initializes foreground as center 60% of image.
    Works well for clinical photos where lesion is centered.
    """
    h, w = image_bgr.shape[:2]
    mask = np.zeros((h, w), np.uint8)

    # Define initial rectangle: center 60% = probable foreground
    margin_x = int(w * 0.2)
    margin_y = int(h * 0.2)
    rect = (margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y)

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(image_bgr, mask, rect, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_RECT)
    except Exception:
        return np.zeros((h, w), np.uint8)

    # Foreground = class 1 or 3
    fg_mask = np.where((mask == 1) | (mask == 3), 255, 0).astype(np.uint8)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=2)

    return fg_mask


def _strategy_kmeans(image_bgr: np.ndarray, k: int = 4) -> np.ndarray:
    """
    K-means color clustering: group pixels into K clusters,
    select the cluster most concentrated near image center (lesion area).
    """
    h, w = image_bgr.shape[:2]
    img_lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)

    # Reshape for kmeans, sample pixels for speed
    pixels = img_lab.reshape(-1, 3).astype(np.float32)

    # Use a subset for speed
    sample_size = min(10000, h * w)
    indices = np.random.choice(h * w, sample_size, replace=False)
    samples = pixels[indices]

    # K-means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    # Create initial labels buffer (required by cv2.kmeans)
    initial_labels = np.zeros(sample_size, dtype=np.int32)
    _, labels, centers = cv2.kmeans(samples, k, initial_labels, criteria, 5, cv2.KMEANS_PP_CENTERS)

    # Assign all pixels to nearest center (on full image)
    # Use batched assignment to avoid OOM
    labels_full = np.zeros(h * w, dtype=np.int32)
    chunk_size = 50000
    for start in range(0, h * w, chunk_size):
        end = min(start + chunk_size, h * w)
        chunk = pixels[start:end]
        dists = np.sum((chunk[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2, axis=2)
        labels_full[start:end] = np.argmin(dists, axis=1)

    labels_2d = labels_full.reshape(h, w)

    # Score each cluster by: centrality (how much is near image center)
    best_cluster = -1
    best_score = -1
    cy_img, cx_img = h // 2, w // 2

    for c in range(k):
        cluster_mask = (labels_2d == c).astype(np.uint8) * 255
        ys, xs = np.where(labels_2d == c)
        if len(ys) < 0.02 * h * w:
            continue  # skip tiny clusters

        # Centrality score: mean distance from cluster pixels to image center
        dists = np.sqrt((xs - cx_img) ** 2 + (ys - cy_img) ** 2)
        centrality = 1.0 / (1.0 + np.mean(dists) / max(w, h))

        # Also check: cluster shouldn't cover >80% of image (background skin)
        coverage = len(ys) / (h * w)
        if coverage > 0.8:
            centrality *= 0.1

        score = centrality

        if score > best_score:
            best_score = score
            best_cluster = c

    if best_cluster < 0:
        return np.zeros((h, w), np.uint8)

    cluster_mask = (labels_2d == best_cluster).astype(np.uint8) * 255

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cluster_mask = cv2.morphologyEx(cluster_mask, cv2.MORPH_CLOSE, kernel)
    cluster_mask = cv2.morphologyEx(cluster_mask, cv2.MORPH_OPEN, kernel)

    return cluster_mask


def _strategy_color_threshold(image_bgr: np.ndarray) -> np.ndarray:
    """
    Multi-color-space thresholding: try LAB, HSV, YCrCb,
    look for regions that differ from the mean skin tone.
    """
    h, w = image_bgr.shape[:2]

    # Try HSV saturation-based (lesions often have different saturation)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    _, S, V = cv2.split(hsv)

    # Otsu on saturation
    _, mask_s = cv2.threshold(S, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Otsu on value (brightness)
    _, mask_v = cv2.threshold(V, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # LAB A-channel (red-green)
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    _, mask_a = cv2.threshold(A, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, mask_l = cv2.threshold(L, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Combine: at least 2 out of 4 agree (voting)
    combined = (
        (mask_s.astype(np.int32) + mask_v.astype(np.int32) +
         mask_a.astype(np.int32) + mask_l.astype(np.int32)) >= 2 * 255
    ).astype(np.uint8) * 255

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)

    return combined


def _strategy_edge_region(image_bgr: np.ndarray) -> np.ndarray:
    """
    Edge-based: detect strong edges, dilate, find largest enclosed region.
    """
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny edge detection
    edges = cv2.Canny(blurred, 30, 100)

    # Dilate to connect nearby edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.dilate(edges, kernel, iterations=3)

    # Close to fill gaps
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=3)

    # Find largest contour
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros((h, w), np.uint8)

    # Draw all large contours as filled
    mask = np.zeros((h, w), np.uint8)
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:3]:
        if cv2.contourArea(cnt) > 0.01 * h * w:
            cv2.drawContours(mask, [cnt], -1, 255, -1)

    return mask


def segment_lesion(image_bgr: np.ndarray) -> tuple:
    """
    Multi-strategy lesion segmentation.

    Tries 4 methods and selects the best result based on:
    - Bbox not covering >85% of image
    - Bbox not too small (<3%)
    - Prefer GrabCut > K-means > Color threshold > Edge

    Returns (mask, bbox_xywh_norm) or (None, None) on failure.
    """
    h, w = image_bgr.shape[:2]

    strategies = [
        ("grabcut", _strategy_grabcut),
        ("kmeans", _strategy_kmeans),
        ("color_thresh", _strategy_color_threshold),
        ("edge", _strategy_edge_region),
    ]

    best_bbox = None
    best_mask = None
    best_score = -1

    for name, strategy_fn in strategies:
        try:
            mask = strategy_fn(image_bgr)
            if mask is None or mask.sum() < 100:
                continue

            bbox = _mask_to_bbox(mask, h, w, min_area_ratio=0.03)
            if bbox is None:
                continue

            _, _, nw, nh = bbox
            bbox_area = nw * nh

            # Score: prefer moderate-sized bboxes (0.05-0.6 of image)
            if 0.05 <= bbox_area <= 0.6:
                score = 3.0
            elif 0.6 < bbox_area <= 0.85:
                score = 1.0
            else:
                score = 0.5

            # Bonus for GrabCut (usually most reliable)
            if name == "grabcut":
                score += 0.5

            if score > best_score:
                best_score = score
                best_bbox = bbox
                best_mask = mask
        except Exception:
            continue

    return best_mask, best_bbox


def process_image(img_path: Path, class_id: int, out_img_dir: Path, out_lbl_dir: Path, img_size: int = 640):
    """
    Process a single image: resize, generate bbox, save image + label.
    Returns True on success.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return False

    h, w = img.shape[:2]
    # Resize to target size (letterboxed by YOLO, but we resize directly here)
    img_resized = cv2.resize(img, (img_size, img_size))
    _, bbox = segment_lesion(img)

    if bbox is None:
        # If all strategies fail, skip this image (don't use noisy label)
        return False

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
    Re-splits data into train/val/test while respecting original split where possible.
    """
    raw_root = Path(raw_root)
    out_root = Path(out_root)

    # Collect all images from train+test dirs
    all_samples = []  # (img_path, class_id, original_split)
    for split in ["train", "test"]:
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
    print(f"Classes: {CLASS_FRIENDLY}")

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
