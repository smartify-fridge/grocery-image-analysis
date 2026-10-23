# Download real-world crops of fridge foods from Open Images V7

import argparse
import csv
import io
import random
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

OUTPUT_ROOT = Path("./open_images")
SEED = 42

CLASS_LIST_URL = "https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv"
IMAGE_URL = "https://open-images-dataset.s3.amazonaws.com/{split}/{image_id}.jpg"

# Open Images split -> (bounding box CSV, our split name)
ANNOTATION_URLS = {
    "train": ("https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv", "train"),
    "validation": ("https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv", "val"),
    "test": ("https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv", "test"),
}

# Open Images class name -> our class name (must match combine_datasets.py)
CLASS_MAP = {
    # Fridge items with no class in the other datasets
    "Egg": "Eggs",
    "Cheese": "Cheese",
    "Bottle": "Bottle",
    "Tin can": "Can",
    "Beer": "Beer",
    "Wine": "Wine",
    "Bread": "Bread",
    # Real-world photos for existing classes
    "Apple": "Apple",
    "Banana": "Banana",
    "Bell pepper": "Bell Pepper",
    "Broccoli": "Broccoli",
    "Cabbage": "Cabbage",
    "Carrot": "Carrot",
    "Coconut": "Coconut",
    "Common fig": "Fig",
    "Cucumber": "Cucumber",
    "Garden Asparagus": "Asparagus",
    "Grape": "Grapes",
    "Grapefruit": "Grapefruit",
    "Lemon (plant)": "Lemon",
    "Mango": "Mango",
    "Milk": "Milk",
    "Mushroom": "Mushroom",
    "Orange (fruit)": "Orange",
    "Peach": "Stone Fruit",
    "Pear": "Pear",
    "Pineapple": "Pineapple",
    "Pomegranate": "Pomegranate",
    "Potato": "Potato",
    "Pumpkin": "Squash",
    "Squash (Plant)": "Squash",
    "Radish": "Radish",
    "Strawberry": "Strawberry",
    "Tomato": "Tomato",
    "Watermelon": "Watermelon",
    "Zucchini": "Zucchini",
}

# Share of each class's crops that go to train/val/test
SPLIT_SHARES = {"train": 0.8, "val": 0.1, "test": 0.1}

MIN_CROP_PIXELS = 48
MAX_CROP_SIDE = 512
MAX_CROPS_PER_IMAGE = 3
CROP_PADDING = 0.08


def open_csv_stream(url):
    """Read a remote CSV row by row without saving it to disk."""
    response = urllib.request.urlopen(url)
    return csv.DictReader(io.TextIOWrapper(response, encoding="utf-8"))


def load_label_ids():
    """Map Open Images label IDs (e.g. /m/014j1m) to our class names."""
    label_ids = {}
    found = set()
    with urllib.request.urlopen(CLASS_LIST_URL) as response:
        for row in csv.reader(io.TextIOWrapper(response, encoding="utf-8")):
            if len(row) >= 2 and row[1] in CLASS_MAP:
                label_ids[row[0]] = CLASS_MAP[row[1]]
                found.add(row[1])

    missing = set(CLASS_MAP) - found
    if missing:
        print(f"[WARNING] Classes not found in Open Images: {sorted(missing)}")
    return label_ids


def collect_boxes(oi_split, label_ids, max_images, rng):
    """
    Stream one split's annotations and pick up to max_images images per class.
    Returns {image_id: [(class_name, xmin, xmax, ymin, ymax), ...]}.
    """
    url, _ = ANNOTATION_URLS[oi_split]
    print(f"Streaming {oi_split} annotations (this can take several minutes)...")

    boxes_by_class = defaultdict(lambda: defaultdict(list))
    for row in open_csv_stream(url):
        class_name = label_ids.get(row["LabelName"])
        # Skip drawings and illustrations of food
        if class_name is None or row["IsDepiction"] == "1":
            continue
        boxes_by_class[class_name][row["ImageID"]].append((
            class_name,
            float(row["XMin"]), float(row["XMax"]),
            float(row["YMin"]), float(row["YMax"]),
        ))

    selected = defaultdict(list)
    for class_name, images in sorted(boxes_by_class.items()):
        image_ids = sorted(images)
        rng.shuffle(image_ids)
        for image_id in image_ids[:max_images]:
            selected[image_id].extend(images[image_id][:MAX_CROPS_PER_IMAGE])
        print(f"  {class_name}: {len(images)} images available, using {min(len(images), max_images)}")
    return selected


def crop_image(oi_split, split, image_id, boxes):
    """Download one image, save its crops, and keep nothing else."""
    try:
        url = IMAGE_URL.format(split=oi_split, image_id=image_id)
        with urllib.request.urlopen(url, timeout=30) as response:
            image = Image.open(io.BytesIO(response.read())).convert("RGB")
    except Exception as error:
        print(f"[WARNING] Could not download {image_id}: {error}")
        return 0

    width, height = image.size
    saved = 0
    for i, (class_name, xmin, xmax, ymin, ymax) in enumerate(boxes):
        x1, x2 = xmin * width, xmax * width
        y1, y2 = ymin * height, ymax * height
        if x2 - x1 < MIN_CROP_PIXELS or y2 - y1 < MIN_CROP_PIXELS:
            continue

        pad_x = (x2 - x1) * CROP_PADDING
        pad_y = (y2 - y1) * CROP_PADDING
        crop = image.crop((
            max(0, int(x1 - pad_x)), max(0, int(y1 - pad_y)),
            min(width, int(x2 + pad_x)), min(height, int(y2 + pad_y)),
        ))
        crop.thumbnail((MAX_CROP_SIDE, MAX_CROP_SIDE))

        destination = OUTPUT_ROOT / split / class_name / f"{image_id}_{i}.jpg"
        destination.parent.mkdir(parents=True, exist_ok=True)
        crop.save(destination, quality=90)
        saved += 1
    return saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--per-class",
        type=int,
        default=300,
        help="Approximate number of images per class across all splits.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Number of parallel image downloads.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=list(ANNOTATION_URLS),
        default=list(ANNOTATION_URLS),
        help="Open Images splits to download (train streams a 2.2GB CSV).",
    )
    args = parser.parse_args()

    rng = random.Random(SEED)
    label_ids = load_label_ids()

    for oi_split in args.splits:
        split = ANNOTATION_URLS[oi_split][1]
        max_images = max(1, round(args.per_class * SPLIT_SHARES[split]))
        selected = collect_boxes(oi_split, label_ids, max_images, rng)

        print(f"Downloading {len(selected)} {oi_split} images...")
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            saved = sum(pool.map(
                lambda item: crop_image(oi_split, split, item[0], item[1]),
                selected.items(),
            ))
        print(f"Saved {saved} {split} crops to {OUTPUT_ROOT / split}")


if __name__ == "__main__":
    main()
