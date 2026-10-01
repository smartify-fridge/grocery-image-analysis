import csv
import os
import zipfile
import numpy as np
import cv2
import kaggle
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

# downloads the Grocery Store dataset from Kaggle if it isn't already on disk
# the zip unpacks to <download_path>/GroceryStoreDataset/dataset/...
def get_data(download_path):
    if not os.path.exists(download_path):
        print("Downloading dataset...")
        kaggle.api.dataset_download_files("validmodel/grocery-store-dataset", path=download_path, unzip=True, quiet=False)
    else:
        print("Dataset already exists. Skipping download.")

# reads classes.csv and returns {class id: class name} for fine or coarse labels
def load_class_names(dataset_path, coarse=False):
    names = {}
    with open(os.path.join(dataset_path, "dataset", "classes.csv"), newline="") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if coarse:
                names[int(row[3])] = row[2]
            else:
                names[int(row[1])] = row[0]
    return names

# loads every image listed in a split file (train.txt / val.txt / test.txt)
# each line is: relative/image/path.jpg, fine label, coarse label
def preprocess(dataset_path, split, im_row, im_col, channels, coarse=False):
    images = []
    labels = []
    dataset_dir = os.path.join(dataset_path, "dataset")

    with open(os.path.join(dataset_dir, f"{split}.txt")) as f:
        for line in f:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue
            image_path = os.path.join(dataset_dir, parts[0])
            label = int(parts[2]) if coarse else int(parts[1])

            if channels == 1:
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            else:
                img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            # skips invalid images/files
            if img is None:
                continue
            if channels == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # cv2.resize takes (width, height)
            img = cv2.resize(img, (im_col, im_row))
            img = img.astype("float32") / 255.0

            images.append(img)
            labels.append(label)

    images = np.array(images, dtype="float32")
    if channels == 1:
        images = np.expand_dims(images, axis=-1)

    return images, np.array(labels)

# prepares data and returns processed train/val/test data with one-hot labels
def prepare_data(dataset_path, im_row, im_col, channels=3, coarse=False):
    label_map = load_class_names(dataset_path, coarse)
    num_classes = max(label_map) + 1

    splits = {}
    for split in ("train", "val", "test"):
        print(f"Loading {split} data...")
        data, labels = preprocess(dataset_path, split, im_row, im_col, channels, coarse)
        splits[split] = (data, to_categorical(labels, num_classes=num_classes))

    return splits["train"], splits["val"], splits["test"], label_map

# downloads Fruits-360 from Kaggle, keeping only the 100x100 images and deleting the 8GB zip after
def get_fruits360(download_path):
    image_dir = os.path.join(download_path, "fruits-360_100x100", "fruits-360")
    if os.path.exists(image_dir):
        print("Fruits-360 already exists. Skipping download.")
        return image_dir

    print("Downloading Fruits-360 (8GB zip, this takes a while)...")
    kaggle.api.dataset_download_files("moltean/fruits", path=download_path, unzip=False, quiet=False)
    zip_path = os.path.join(download_path, "fruits.zip")
    print("Extracting 100x100 images...")
    with zipfile.ZipFile(zip_path) as z:
        members = [m for m in z.namelist() if m.startswith("fruits-360_100x100/")]
        z.extractall(download_path, members=members)
    os.remove(zip_path)
    return image_dir

# streams Fruits-360 from disk in batches, since it's too big to hold in memory like the grocery data
def prepare_fruits360(image_dir, im_row, im_col, batch_size):
    datasets = []
    for split in ("Training", "Test"):
        ds = tf.keras.utils.image_dataset_from_directory(
            os.path.join(image_dir, split),
            label_mode="categorical",
            image_size=(im_row, im_col),
            batch_size=batch_size,
            shuffle=(split == "Training"),
        )
        num_classes = len(ds.class_names)
        # scale to [0, 1] to match the grocery images
        ds = ds.map(lambda x, y: (x / 255.0, y)).prefetch(tf.data.AUTOTUNE)
        datasets.append(ds)
    return datasets[0], datasets[1], num_classes
