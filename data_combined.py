import json
from pathlib import Path
import cv2
import numpy as np
import tensorflow as tf

DATASET_ROOT = Path("./combined_dataset")
CLASS_NAMES_PATH = Path("./class_names.json")

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

AUTOTUNE = tf.data.AUTOTUNE

def get_class_names(dataset_root=DATASET_ROOT):
    """
    Generate one deterministic class order from the training folders.

    Expected structure:
        combined_dataset/
            train/
                Apple/
                Banana/
            val/
                Apple/
                Banana/
            test/
                Apple/
                Banana/

    Classes are sorted alphabetically and assigned indices starting
    at zero. Validation/test classes must exist in the training set.
    """

    dataset_root = Path(dataset_root)
    train_dir = dataset_root / "train"

    if not train_dir.is_dir():
        raise FileNotFoundError(
            f"Training directory not found: {train_dir.resolve()}"
        )

    class_names = sorted(
        path.name
        for path in train_dir.iterdir()
        if path.is_dir()
    )

    if not class_names:
        raise ValueError(
            f"No class folders found in {train_dir.resolve()}"
        )

    # Check that each split contains only classes known to training.
    for split in ("val", "test"):
        split_dir = dataset_root / split

        if not split_dir.is_dir():
            raise FileNotFoundError(
                f"Dataset split not found: {split_dir.resolve()}"
            )

        split_classes = {
            path.name
            for path in split_dir.iterdir()
            if path.is_dir()
        }

        unknown_classes = split_classes - set(class_names)

        if unknown_classes:
            raise ValueError(
                f"Classes found in {split}/ but not in train/: "
                f"{sorted(unknown_classes)}. "
                "Move or correct these folders before training."
            )

    return class_names


def save_class_names(
    class_names,
    output_path=CLASS_NAMES_PATH,
):
    """
    Save the exact class order used by the classifier.

    The JSON is a list, where the list index is the class index.
    This format is compatible with use_model.py.
    """

    output_path = Path(output_path)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            class_names,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Saved class mapping: {output_path.resolve()}")
    print(f"Number of classes: {len(class_names)}")


def get_image_paths(split_dir, class_names):
    """
    Collect supported image files and their integer class labels.

    Returns:
        image_paths: list of image paths
        image_labels: list of integer class indices
        class_counts: dictionary of image counts per class
    """

    split_dir = Path(split_dir)

    label_map = {
        class_name: index
        for index, class_name in enumerate(class_names)
    }

    image_paths = []
    image_labels = []

    class_counts = {
        class_name: 0
        for class_name in class_names
    }

    for class_name in class_names:
        class_dir = split_dir / class_name

        # A class may be absent from val/test, but not from train.
        if not class_dir.is_dir():
            continue

        files = sorted(
            path
            for path in class_dir.rglob("*")
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        )

        for image_path in files:
            image_paths.append(str(image_path))
            image_labels.append(label_map[class_name])
            class_counts[class_name] += 1

    return image_paths, image_labels, class_counts


def load_and_preprocess_image(
    image_path,
    img_rows,
    img_cols,
):
    """
    Read an image using OpenCV and prepare it for MobileNet/LeNet.

    Processing:
        1. Read image in BGR format.
        2. Convert BGR to RGB.
        3. Resize to the requested dimensions.
        4. Convert to float32.
        5. Normalize pixel values to [0, 1].
    """

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(
            f"Could not read image: {image_path}"
        )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    )

    image = cv2.resize(
        image,
        (img_cols, img_rows),
        interpolation=cv2.INTER_LINEAR,
    )

    image = image.astype(np.float32) / 255.0

    return image

def create_tf_dataset(
    image_paths,
    image_labels,
    class_count,
    img_rows,
    img_cols,
    batch_size,
    shuffle=False,
):
    """
    Create a memory-efficient tf.data.Dataset.

    Images are loaded lazily instead of loading the entire dataset
    into RAM. Labels are one-hot encoded for categorical
    classification.
    """

    if len(image_paths) == 0:
        raise ValueError(
            "No supported images were found in this dataset split."
        )

    if len(image_paths) != len(image_labels):
        raise ValueError(
            "Image path count does not match image label count."
        )

    image_paths = list(image_paths)
    image_labels = list(image_labels)
    rng = np.random.default_rng()

    def generator():
        # Paths are listed class by class, so shuffle the whole list each epoch
        # (a small shuffle buffer alone only mixes a few neighboring classes).
        order = (
            rng.permutation(len(image_paths))
            if shuffle
            else range(len(image_paths))
        )
        for index in order:
            image_path = image_paths[index]
            label = image_labels[index]
            image = load_and_preprocess_image(
                image_path,
                img_rows,
                img_cols,
            )

            one_hot_label = np.zeros(
                class_count,
                dtype=np.float32,
            )

            one_hot_label[label] = 1.0

            yield image, one_hot_label

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(
                shape=(img_rows, img_cols, 3),
                dtype=tf.float32,
            ),
            tf.TensorSpec(
                shape=(class_count,),
                dtype=tf.float32,
            ),
        ),
    )

    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.apply(
        tf.data.experimental.assert_cardinality(
            (len(image_paths) + batch_size - 1) // batch_size
        )
    )
    dataset = dataset.prefetch(AUTOTUNE)

    return dataset


def prepare_combined_data(
    img_rows,
    img_cols,
    batch_size=32,
    dataset_root=DATASET_ROOT,
    class_names_path=CLASS_NAMES_PATH,
):
    """
    Prepare train, validation, and test datasets.

    Args:
        img_rows: Image height (192 for MobileNet, 64 for LeNet).
        img_cols: Image width (192 for MobileNet, 64 for LeNet).
        batch_size: Number of images per batch.
        dataset_root: Root directory containing train/val/test.
        class_names_path: Destination for class_names.json.

    Returns:
        train_dataset
        val_dataset
        test_dataset
        label_map: {class_name: integer_index}
        train_counts: {class_name: number_of_training_images}
    """

    dataset_root = Path(dataset_root)

    if img_rows <= 0 or img_cols <= 0:
        raise ValueError(
            "Image dimensions must be positive."
        )

    if batch_size <= 0:
        raise ValueError(
            "Batch size must be positive."
        )

    print("=" * 60)
    print("Preparing combined grocery dataset")
    print("=" * 60)

    class_names = get_class_names(dataset_root)

    save_class_names(
        class_names,
        class_names_path,
    )

    label_map = {
        class_name: index
        for index, class_name in enumerate(class_names)
    }

    num_classes = len(class_names)

    print(f"\nNumber of classes: {num_classes}")
    print(f"Image size: {img_rows} x {img_cols}")
    print(f"Batch size: {batch_size}")

    train_paths, train_labels, train_counts = get_image_paths(
        dataset_root / "train",
        class_names,
    )

    val_paths, val_labels, val_counts = get_image_paths(
        dataset_root / "val",
        class_names,
    )

    test_paths, test_labels, test_counts = get_image_paths(
        dataset_root / "test",
        class_names,
    )

    if not train_paths:
        raise ValueError(
            "The training split contains no supported images."
        )

    if not val_paths:
        raise ValueError(
            "The validation split contains no supported images."
        )

    if not test_paths:
        raise ValueError(
            "The test split contains no supported images."
        )

    empty_training_classes = [
        name
        for name, count in train_counts.items()
        if count == 0
    ]

    if empty_training_classes:
        raise ValueError(
            "These class folders have no supported training images: "
            f"{empty_training_classes}"
        )

    print("\nDataset image counts:")
    print(f"  Training:   {len(train_paths)}")
    print(f"  Validation: {len(val_paths)}")
    print(f"  Test:       {len(test_paths)}")
    print(
        "  Total:      "
        f"{len(train_paths) + len(val_paths) + len(test_paths)}"
    )

    print("\nClass mapping preview:")
    for index, class_name in enumerate(class_names[:10]):
        print(f"  {index}: {class_name}")

    if num_classes > 10:
        print(f"  ... and {num_classes - 10} more classes")

    train_dataset = create_tf_dataset(
        train_paths,
        train_labels,
        num_classes,
        img_rows,
        img_cols,
        batch_size,
        shuffle=True,
    )

    val_dataset = create_tf_dataset(
        val_paths,
        val_labels,
        num_classes,
        img_rows,
        img_cols,
        batch_size,
        shuffle=False,
    )

    test_dataset = create_tf_dataset(
        test_paths,
        test_labels,
        num_classes,
        img_rows,
        img_cols,
        batch_size,
        shuffle=False,
    )

    print("\nDatasets are ready.")
    print("=" * 60)

    return (
        train_dataset,
        val_dataset,
        test_dataset,
        label_map,
        train_counts,
    )