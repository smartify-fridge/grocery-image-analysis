# Combine 4 datasets to one large datasets

import argparse
import csv
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

GROCERY_ROOT = Path("./grocery_kaggle/GroceryStoreDataset")
FRUITS_ROOT = Path("./fruits360_kaggle/fruits-360_100x100/fruits-360")
INGREDIENT_ROOT = Path("./ingredient_dataset")
FOOD_INGREDIENT_51_ROOT = Path("./food_ingredient_51")
OUTPUT_ROOT = Path("./combined_dataset")

SEED = 42
IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}

# Explicit aliases for ingredients that refer to the same class.
ALIASES = {
    "bell pepper": "Bell Pepper",
    "capsicum": "Bell Pepper",
    "chilli pepper": "Chili Pepper",
    "chilli": "Chili Pepper",
    "chili": "Chili Pepper",
    "jalepeno": "Jalapeno",
    "dragon_fruit": "Dragon Fruit",
    "dragon fruit": "Dragon Fruit",
    "raddish": "Radish",
    "radish": "Radish",
    "sweetcorn": "Corn",
    "sweet corn": "Corn",
    "soy beans": "Soybeans",
    "soybean": "Soybeans",
    "spiny groud": "Spiny Gourd",
    "spiny gourd": "Spiny Gourd",
}

# Broad fridge-level classes and the canonical labels merged into each.
# Labels not listed here keep their own name as a class.
BROAD_CLASSES = {
    # Fruit
    "Apple": ["Apple Braeburn", "Apple Granny Smith", "Pink-Lady", "Royal-Gala"],
    "Pear": ["Pear", "Conference", "Kaiser"],
    "Mandarin": ["Clementine", "Mandarine", "Satsumas", "Tangelo"],
    "Grapefruit": ["Red-Grapefruit", "Pomelo Sweetie"],
    "Berries": ["Blueberry", "Raspberry", "Blackberry", "Mulberry", "Redcurrant"],
    "Stone Fruit": ["Peach", "Nectarine", "Apricot", "Plum"],
    "Melon": ["Cantaloupe", "Galia-Melon", "Honeydew-Melon"],
    "Dragon Fruit": ["Dragon Fruit", "Pitahaya Red"],
    "Passion Fruit": ["Passion Fruit", "Maracuja"],
    # Vegetables
    "Tomato": ["Tomato", "Regular-Tomato", "Beef-Tomato", "Vine-Tomato", "Tomato Cherry"],
    "Bell Pepper": ["Bell Pepper", "Orange-Bell-Pepper", "Paprika"],
    "Chili Pepper": ["Chili Pepper", "Jalapeno"],
    "Potato": ["Potato White", "Floury-Potato", "Solid-Potato"],
    "Sweet Potato": ["Sweet-Potato"],
    "Onion": ["Onion Red", "Yellow-Onion"],
    "Carrot": ["Carrots"],
    "Beet": ["Beetroot", "Red-Beet"],
    "Mushroom": ["Brown-Cap-Mushroom", "Mushroom Champignon"],
    "Squash": ["Squash", "Pumpkin"],
    "Leafy Greens": ["Spinach", "Kale", "Chard", "Amaranth"],
    "Fresh Herbs": ["Basil", "Cilantro", "Dill", "Mint", "Oregano", "Parsley", "Rosemary", "Thyme"],
    "Beans & Peas": ["Bean Pod", "Peas", "Soybeans"],
    "Other Vegetable": [
        "Fennel", "Okra", "Parsnip", "Turnip", "Kohlrabi", "Bitter Gourd",
        "Bottle Gourd", "Ridge Gourd", "Spiny Gourd", "Sponge Gourd",
    ],
    # Meat and seafood
    "Red Meat": ["Beef Steak", "Veal Cutlet", "Pork Chop", "Lamb Shank", "Goat Meat"],
    "Poultry": ["Chicken Breast", "Turkey Breast", "Duck Breast"],
    "Bacon & Deli Meat": ["Bacon", "Ham", "Prosciutto"],
    "Fish": [
        "Cod Fillet", "Herring Fillet", "Mackerel Fillet",
        "Salmon Fillet", "Sardine Fillet", "Tuna Steak",
    ],
    "Shellfish & Seafood": [
        "Clam Meat", "Crab Meat", "Lobster Tail", "Scallop",
        "Shrimp", "Octopus", "Squid", "Caviar",
    ],
    # Dairy and drinks
    "Milk": [
        "Arla-Ecological-Medium-Fat-Milk", "Arla-Lactose-Medium-Fat-Milk",
        "Arla-Medium-Fat-Milk", "Arla-Standard-Milk", "Arla-Sour-Milk",
        "Garant-Ecological-Medium-Fat-Milk", "Garant-Ecological-Standard-Milk",
    ],
    "Plant Milk": ["Alpro-Fresh-Soy-Milk", "Alpro-Shelf-Soy-Milk", "Oatly-Oat-Milk"],
    "Yogurt": [
        "Alpro-Blueberry-Soyghurt", "Alpro-Vanilla-Soyghurt",
        "Arla-Mild-Vanilla-Yoghurt", "Arla-Natural-Mild-Low-Fat-Yoghurt",
        "Arla-Natural-Yoghurt", "Oatly-Natural-Oatghurt", "Valio-Vanilla-Yoghurt",
        "Yoggi-Strawberry-Yoghurt", "Yoggi-Vanilla-Yoghurt",
    ],
    "Sour Cream": ["Arla-Sour-Cream", "Arla-Ecological-Sour-Cream"],
    "Juice": [
        "Bravo-Apple-Juice", "Bravo-Orange-Juice", "God-Morgon-Apple-Juice",
        "God-Morgon-Orange-Juice", "God-Morgon-Orange-Red-Grapefruit-Juice",
        "God-Morgon-Red-Grapefruit-Juice", "Tropicana-Apple-Juice",
        "Tropicana-Golden-Grapefruit", "Tropicana-Juice-Smooth",
        "Tropicana-Mandarin-Morning",
    ],
}

LABEL_TO_BROAD = {
    label: broad
    for broad, labels in BROAD_CLASSES.items()
    for label in labels
}

# Pantry items that are not kept in a fridge.
DROPPED_CLASSES = {
    "Cardamom", "Cinnamon", "Clove", "Cumin", "Nutmeg", "Turmeric", "Honey",
    "Almonds", "Caju Seed", "Chestnut", "Hazelnut", "Nut", "Pistachio", "Walnut",
}

SPLIT_NAMES = {
    "train": "train",
    "training": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "test": "test",
    "testing": "test",
}

def canonical_label(label):
    """Normalize spelling/case and apply explicit aliases."""
    label = " ".join(label.replace("_", " ").strip().split())
    return ALIASES.get(label.casefold(), label.title())

def apply_broad_classes(records):
    """Drop pantry classes and merge the rest into broad classes."""
    kept = []
    for record in records:
        if record["label"] in DROPPED_CLASSES:
            continue
        record["label"] = LABEL_TO_BROAD.get(record["label"], record["label"])
        kept.append(record)
    print(f"\nDropped {len(records) - len(kept)} pantry images")
    return kept

def image_files(folder):
    if not folder.exists():
        return []
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

def split_label_from_path(path, root):
    """
    Find a recognized split folder and the class folder immediately
    beneath it. Returns (split, label), or (None, class_folder).
    """
    relative_parts = path.relative_to(root).parts

    for i, part in enumerate(relative_parts[:-1]):
        split = SPLIT_NAMES.get(part.casefold())
        if split and i + 1 < len(relative_parts) - 1:
            return split, relative_parts[i + 1]

    # A dataset with class folders directly under its root.
    if len(relative_parts) >= 2:
        return None, relative_parts[0]

    return None, None


def collect_class_folder_dataset(root, source):
    records = []

    if not root.exists():
        print(f"[WARNING] Missing {source} directory: {root}")
        return records

    for path in image_files(root):
        split, label = split_label_from_path(path, root)
        if label:
            records.append({
                "path": path,
                "label": canonical_label(label),
                "split": split,
                "source": source,
            })

    print(f"{source}: found {len(records)} images")
    return records


def collect_grocery_dataset():
    """
    Read Grocery Store Dataset split manifests.
    Manifest rows usually contain a relative image path and class ID.
    """
    records = []
    dataset_dir = GROCERY_ROOT / "dataset"
    classes_file = dataset_dir / "classes.csv"

    if not classes_file.exists():
        print(f"[WARNING] Missing Grocery classes file: {classes_file}")
        return records

    # Support either ID,name or name,ID CSV column ordering.
    class_names = {}
    with classes_file.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))

    for row in rows:
        if len(row) < 2:
            continue
        a, b = row[0].strip(), row[1].strip()
        if a.isdigit():
            class_names[a] = b
        elif b.isdigit():
            class_names[b] = a

    for split in ("train", "val", "test"):
        manifest = dataset_dir / f"{split}.txt"
        if not manifest.exists():
            print(f"[WARNING] Missing manifest: {manifest}")
            continue

        with manifest.open("r", encoding="utf-8-sig") as f:
            for line in f:
                
                line = line.strip()
                if not line:
                    continue

                parts = [part.strip() for part in line.split(",") if part.strip()]

                if len(parts) < 2:
                    # Fallback for whitespace-separated manifests.
                    parts = line.rsplit(maxsplit=1)

                if len(parts) < 2:
                    print(f"[WARNING] Cannot parse manifest line: {line}")
                    continue

                relative_path = parts[0]
                class_id = parts[1].strip()

                label = class_names.get(class_id)

                if label is None and not class_id.isdigit():
                    label = class_id

                if not label:
                    print(f"[WARNING] Unknown class ID: {class_id}")
                    continue

                # Try paths relative to the dataset root and dataset metadata directory.
                candidates = [
                    GROCERY_ROOT / relative_path,
                    GROCERY_ROOT / "dataset" / relative_path,
                ]

                image_path = next((p for p in candidates if p.is_file()), None)

                if image_path is None:
                    print(f"[WARNING] Image not found: {relative_path}")
                    continue

                records.append({
                    "path": image_path,
                    "label": canonical_label(label),
                    "split": split,
                    "source": "Grocery Store",
                })

    print(f"Grocery Store: found {len(records)} images")
    return records

def assign_splits(records, rng):
    """
    Split each source/class independently.

    Existing train/val/test assignments are retained when possible.
    If only train + test exist, carve validation from train.
    If only train + val exist, carve test from train.
    If there are no predefined splits, use 80/10/10.
    """
    groups = defaultdict(list)
    for record in records:
        groups[(record["source"], record["label"])].append(record)

    result = []

    for (source, label), items in groups.items():
        rng.shuffle(items)

        predefined = [r for r in items if r["split"] is not None]
        unassigned = [r for r in items if r["split"] is None]
        existing = {r["split"] for r in predefined}

        # A class-folder-only dataset: split it 80/10/10.
        if not predefined:
            n = len(items)
            n_test = max(1, round(n * 0.10)) if n >= 3 else 0
            n_val = max(1, round(n * 0.10)) if n - n_test >= 3 else 0
            for i, r in enumerate(items):
                r["split"] = (
                    "test" if i < n_test else
                    "val" if i < n_test + n_val else
                    "train"
                )
            result.extend(items)
            continue

        result.extend(predefined)

        # Use unassigned images as training data by default.
        for r in unassigned:
            r["split"] = "train"
        result.extend(unassigned)

        # Split definitions missing validation or test are handled below.
        train_items = [r for r in items if r["split"] == "train"]
        rng.shuffle(train_items)

        if "val" not in existing and len(train_items) >= 3:
            count = max(1, round(len(train_items) * 0.10))
            for r in train_items[:count]:
                r["split"] = "val"

        train_items = [r for r in items if r["split"] == "train"]
        rng.shuffle(train_items)

        if "test" not in existing and len(train_items) >= 3:
            count = max(1, round(len(train_items) * 0.10))
            for r in train_items[:count]:
                r["split"] = "test"

    return result


def copy_dataset(records, overwrite=False):
    if OUTPUT_ROOT.exists():
        if not overwrite:
            raise FileExistsError(
                f"{OUTPUT_ROOT} already exists. "
                "Use --overwrite to rebuild it."
            )
        shutil.rmtree(OUTPUT_ROOT)

    for split in ("train", "val", "test"):
        (OUTPUT_ROOT / split).mkdir(parents=True, exist_ok=True)

    # Give copied images unique filenames to avoid collisions.
    counters = Counter()
    class_counts = {s: Counter() for s in ("train", "val", "test")}

    for record in records:
        split = record["split"]
        label = record["label"]
        source_path = record["path"]

        if split not in class_counts:
            continue

        destination_dir = OUTPUT_ROOT / split / label
        destination_dir.mkdir(parents=True, exist_ok=True)

        counters[(split, label)] += 1
        index = counters[(split, label)]
        destination = destination_dir / (
            f"{record['source'].replace(' ', '_')}_{index:06d}"
            f"{source_path.suffix.lower()}"
        )

        shutil.copy2(source_path, destination)
        class_counts[split][label] += 1

    print("\nDataset created successfully.")
    print(f"Output: {OUTPUT_ROOT.resolve()}")

    for split in ("train", "val", "test"):
        counts = class_counts[split]
        print(
            f"{split}: {sum(counts.values())} images, "
            f"{len(counts)} classes"
        )

    all_classes = set().union(*(set(c) for c in class_counts.values()))
    missing = {
        label: [s for s in ("train", "val", "test")
                if class_counts[s][label] == 0]
        for label in all_classes
    }
    missing = {k: v for k, v in missing.items() if v}

    if missing:
        print("\n[WARNING] Some classes are missing from one or more splits:")
        for label, splits in sorted(missing.items()):
            print(f"  {label}: missing {', '.join(splits)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete and rebuild the existing combined_dataset folder.",
    )
    args = parser.parse_args()

    rng = random.Random(SEED)

    records = []
    records.extend(collect_grocery_dataset())
    records.extend(
        collect_class_folder_dataset(FRUITS_ROOT, "Fruits-360")
    )
    records.extend(
        collect_class_folder_dataset(INGREDIENT_ROOT, "Ingredient Dataset")
    )
    records.extend(
        collect_class_folder_dataset(
            FOOD_INGREDIENT_51_ROOT, "Food Ingredient 51"
        )
    )

    if not records:
        raise RuntimeError(
            "No images found. Check the dataset paths at the top of this file."
        )

    records = apply_broad_classes(records)
    records = assign_splits(records, rng)

    # Remove exact duplicate files by content hash, keeping one copy.
    # This also prevents identical files from appearing in multiple splits.
    import hashlib

    seen_hashes = set()
    unique_records = []
    for record in records:
        digest = hashlib.sha256(record["path"].read_bytes()).hexdigest()
        if digest in seen_hashes:
            continue
        seen_hashes.add(digest)
        unique_records.append(record)

    print(f"\nUnique images to copy: {len(unique_records)}")
    copy_dataset(unique_records, overwrite=args.overwrite)

if __name__ == "__main__":
    main()