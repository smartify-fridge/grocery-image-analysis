"""
Crops labeled items out of a Roboflow COCO export into combined_dataset/ class folders.

Run:
    python import_detection_crops.py --zip "~/Downloads/FoodTypes - individual - in fridge.v1i.coco.zip" --source FoodTypes --dry-run
    python import_detection_crops.py --zip "~/Downloads/FoodTypes - individual - in fridge.v1i.coco.zip" --source FoodTypes

Imported so far (credit in reports):
    FoodTypes   https://universe.roboflow.com/tk-matima-unqyz/foodtypes-individual-in-fridge  (CC BY 4.0)
Tried and dropped:
    FridgeFoodDetection (MIT): Roboflow's noise and black cutout squares are baked into most images,
    and it's photos of a single fridge.

The export's train/valid/test splits map to train/val/test, so no crop leaks into another split.
Roboflow's augmented copies of a photo (same name before ".rf.") are counted once.
Re-running with the same --source replaces that source's earlier crops.
"""
import argparse
import io
import json
import random
import zipfile
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageOps

DATASET_ROOT = Path("combined_dataset")
SPLITS = {"train": "train", "valid": "val", "test": "test"}
MIN_BOX_SIDE = 40
MAX_CROP_SIDE = 384
PADDING = 0.05

# source label (lowercase) -> one of the 86 classes; labels not listed are skipped
COMMON = {
    "apple": "Apple", "apples": "Apple", "avocado": "Avocado", "banana": "Banana", "bananas": "Banana",
    "beef": "Red Meat", "ground_beef": "Red Meat", "meat": "Red Meat",
    "ham": "Bacon & Deli Meat", "bacon": "Bacon & Deli Meat", "salami": "Bacon & Deli Meat",
    "sausage": "Sausage", "chicken": "Poultry", "chicken_breast": "Poultry",
    "shrimp": "Shellfish & Seafood", "fish": "Fish",
    "egg": "Eggs", "eggs": "Eggs",
    "milk": "Milk", "heavy_cream": "Milk", "cream": "Milk", "fresh cream": "Milk",
    "yoghurt": "Yogurt", "yogurt": "Yogurt", "juice": "Juice",
    "cheese": "Cheese", "goat_cheese": "Cheese", "cream cheese": "Cheese",
    "bread": "Bread", "bazlama": "Bread",
    "tomato": "Tomato", "tomatoes": "Tomato", "potato": "Potato", "potatoes": "Potato",
    "sweet_potato": "Sweet Potato", "carrot": "Carrot", "carrots": "Carrot",
    "onion": "Onion", "red onion": "Onion", "spring onion": "Onion",
    "corn": "Corn", "sweetcorn": "Corn", "green_beans": "Beans & Peas", "green beans": "Beans & Peas", "peas": "Beans & Peas",
    "mushroom": "Mushroom", "mushrooms": "Mushroom",
    "spinach": "Leafy Greens", "green leaves": "Leafy Greens", "salad": "Leafy Greens", "lettuce": "Lettuce",
    "cabbage": "Cabbage", "broccoli": "Broccoli", "cauliflower": "Cauliflower", "asparagus": "Asparagus",
    "cucumber": "Cucumber", "zucchini": "Zucchini", "courgettes": "Zucchini", "squash": "Squash",
    "eggplant": "Eggplant", "aubergine": "Eggplant", "beetroot": "Beet", "ginger": "Ginger", "parsley": "Fresh Herbs",
    "capsicum": "Bell Pepper", "peppers": "Bell Pepper", "red bell pepper": "Bell Pepper",
    "green bell pepper": "Bell Pepper", "yellow bell pepper": "Bell Pepper",
    "green chilly": "Chili Pepper", "green chilies": "Chili Pepper",
    "strawberry": "Strawberry", "strawberries": "Strawberry",
    "blueberry": "Berries", "blueberries": "Berries", "raspberries": "Berries", "blackberries": "Berries",
    "lemon": "Lemon", "lemons": "Lemon", "lime": "Lime", "orange": "Orange", "oranges": "Orange",
    "clementine": "Mandarin", "grapes": "Grapes", "red grapes": "Grapes",
    "nectarine": "Stone Fruit", "plum": "Stone Fruit", "peach": "Stone Fruit",
    "cantaloupe": "Melon", "galia": "Melon", "honeydew": "Melon", "watermelon": "Watermelon",
    "mango": "Mango", "dates": "Dates", "pineapple": "Pineapple",
}


def original_name(file_name):
    return file_name.split(".rf.")[0]


def load_split(z, prefix, split):
    data = json.loads(z.read(f"{prefix}{split}/_annotations.coco.json"))
    categories = {c["id"]: c["name"] for c in data["categories"]}
    images = {i["id"]: i for i in data["images"]}

    # one Roboflow copy per original photo
    keep, seen = set(), set()
    for image in sorted(images.values(), key=lambda i: i["file_name"]):
        if original_name(image["file_name"]) not in seen:
            seen.add(original_name(image["file_name"]))
            keep.add(image["id"])

    boxes = defaultdict(list)
    for a in data["annotations"]:
        label = categories[a["category_id"]].strip().lower()
        cls = COMMON.get(label)
        x, y, w, h = a["bbox"]
        if cls and a["image_id"] in keep and min(w, h) >= MIN_BOX_SIDE:
            boxes[cls].append((f"{prefix}{split}/{images[a['image_id']]['file_name']}", (x, y, w, h)))
    return boxes


def crop(z, path, box):
    image = ImageOps.exif_transpose(Image.open(io.BytesIO(z.read(path)))).convert("RGB")
    x, y, w, h = box
    pad_x, pad_y = w * PADDING, h * PADDING
    region = image.crop((
        max(0, int(x - pad_x)), max(0, int(y - pad_y)),
        min(image.width, int(x + w + pad_x)), min(image.height, int(y + h + pad_y)),
    ))
    region.thumbnail((MAX_CROP_SIDE, MAX_CROP_SIDE))
    return region


def main():
    parser = argparse.ArgumentParser(description="Import Roboflow COCO crops into combined_dataset/.")
    parser.add_argument("--zip", required=True, help="Roboflow COCO export zip")
    parser.add_argument("--source", required=True, help="file name prefix, e.g. FoodTypes")
    parser.add_argument("--cap", type=int, default=250, help="max train crops per class (val/test get cap // 6)")
    parser.add_argument("--dry-run", action="store_true", help="only print how many crops each class would get")
    args = parser.parse_args()

    class_names = set(json.loads(Path("trained_model/class_names.json").read_text()))
    assert set(COMMON.values()) <= class_names, set(COMMON.values()) - class_names

    z = zipfile.ZipFile(Path(args.zip).expanduser())
    prefix = next(n for n in z.namelist() if n.endswith("train/_annotations.coco.json"))[: -len("train/_annotations.coco.json")]
    rng = random.Random(0)
    plan = {}
    for split, target in SPLITS.items():
        cap = args.cap if target == "train" else max(1, args.cap // 6)
        for cls, boxes in load_split(z, prefix, split).items():
            rng.shuffle(boxes)
            plan[(target, cls)] = boxes[:cap]

    for cls in sorted({c for _, c in plan}):
        counts = [len(plan.get((t, cls), [])) for t in SPLITS.values()]
        existing = len(list((DATASET_ROOT / "train" / cls).glob("*")))
        print(f"  {cls:22} +{counts[0]:4} train  +{counts[1]:3} val  +{counts[2]:3} test   (train now {existing})")
    print(f"Total: {sum(len(v) for v in plan.values())} crops")
    if args.dry_run:
        return

    for old in DATASET_ROOT.glob(f"*/*/{args.source}_*"):
        old.unlink()
    for (target, cls), boxes in plan.items():
        folder = DATASET_ROOT / target / cls
        folder.mkdir(parents=True, exist_ok=True)
        for n, (path, box) in enumerate(boxes, 1):
            crop(z, path, box).save(folder / f"{args.source}_{target}_{n:05d}.jpg", quality=90)
    print(f"Saved under {DATASET_ROOT}/<split>/<class>/{args.source}_*.jpg")


if __name__ == "__main__":
    main()
