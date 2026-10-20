# download ingreident & food_ingredent_51 datasets
import os
from collections import Counter
import kagglehub

# Supported image file types
IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}

# The two additional ingredient image datasets
INGREDIENT_DATASETS = {
    "ingredient": "keno31/ingredient-dataset",
    "food_ingredient_51": "sunnyagarwal427444/food-ingredient-dataset-51",
}

def get_ingredient_datasets():
    """
    Download both datasets if necessary.
    KaggleHub reuses its cache when the dataset is already downloaded.

    Returns:
        Dictionary mapping dataset names to local dataset paths.
    """
    paths = {}

    for name, dataset_id in INGREDIENT_DATASETS.items():
        print(f"\nDownloading/checking {dataset_id}...")

        path = kagglehub.dataset_download(dataset_id)
        paths[name] = path

        print(f"{name} dataset path: {path}")

    return paths


def inspect_dataset(path, max_depth=3):
    """
    Display the dataset folder structure and image counts.

    Prints:
        - Relative folder paths
        - Number of subfolders
        - Number of image files in each displayed folder
        - Total number of images
        - Image counts for likely class folders
    """
    print(f"\nInspecting dataset: {path}")

    if not os.path.isdir(path):
        print("ERROR: Dataset directory does not exist.")
        return

    total_images = 0
    class_counts = Counter()

    for root, dirs, files in os.walk(path):
        # Skip common hidden/cache directories
        dirs[:] = sorted(
            d for d in dirs
            if not d.startswith(".")
        )
        files.sort()

        relative_path = os.path.relpath(root, path)
        depth = 0 if relative_path == "." else relative_path.count(os.sep) + 1

        image_files = [
            filename for filename in files
            if os.path.splitext(filename)[1].lower()
            in IMAGE_EXTENSIONS
        ]

        total_images += len(image_files)

        # A class folder typically contains images directly.
        if image_files:
            class_counts[relative_path] = len(image_files)

        if depth <= max_depth:
            print(
                f"{relative_path} | "
                f"subfolders={len(dirs)} | "
                f"images={len(image_files)}"
            )

        # Limit displayed tree depth, but continue counting all images.
        if depth >= max_depth:
            dirs[:] = []

    print(f"\nTotal images found: {total_images}")

    if class_counts:
        print("\nFolders containing images:")
        for folder, count in sorted(class_counts.items()):
            print(f"  {folder}: {count} images")

def main():
    dataset_paths = get_ingredient_datasets()

    for name, path in dataset_paths.items():
        print("\n" + "=" * 60)
        print(f"DATASET: {name}")
        print("=" * 60)

        inspect_dataset(path, max_depth=3)

if __name__ == "__main__":
    main()