import numpy as np
import os
import kaggle
import cv2
from tensorflow.keras.utils import to_categorical

# checks if data set path exists and downloads it if path doesn't exist yet
# prints a statement to update user regarding status
def get_data(dataset_path):
    if not os.path.exists(dataset_path):
        print("Downloading dataset...")
        # downloads dataset from kaggle database and unzips in path folder
        kaggle.api.dataset_download_files("techsash/waste-classification-data", path=dataset_path, unzip=True)
    else:
        print("Dataset already exists. Skipping download.")

# preprocess the images so they are uniform size and in greyscale
def preprocess(dir, im_row, im_col):
    images = []
    labels = []
    
    # iterating through each directory and checking for the image
    for label in os.listdir(dir): 
        label_dir = os.path.join(dir, label)
        if os.path.isdir(label_dir):
            for image_name in os.listdir(label_dir):
                image_path = os.path.join(label_dir, image_name)
                # Load the image as grayscale to be compatible with LeNet
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  
                # skips inavlid images/files
                if img is None:
                    continue  
                # resize to be compatible with LeNet
                img = cv2.resize(img, (28,28))
                
                # normalize pixel values to [0, 1] greyscale
                img = img.astype("float32") / 255.0
                
                # append to list
                images.append(img)
                labels.append(label)
    
    # convert lists to numpy arrays
    images = np.array(images, dtype="float32")
    labels = np.array(labels)

    # reshape images to match LeNet input shape (28, 28, 1)
    images = np.expand_dims(images, axis=-1)

    return images, labels

# converts labels to onehot encoding
def labels_to_onehot(labels:list, label_map:dict):
    # iterates through each label and changes it to numpy array 
    labels = np.array([label_map[label] for label in labels])
    # keras convert from numpy to onehot encoding
    return to_categorical(labels, num_classes=len(label_map))


# prepares data and return processed training and testing data with labels
def prepare_data(train_dir, test_dir, im_row, im_col):
    print("Loading training data...")
    train_data, train_labels = preprocess(train_dir, im_row, im_col)
    test_data, test_labels = preprocess(test_dir, im_row, im_col)
    
    # convert labels to integers
    class_names = sorted(os.listdir(train_dir))
    label_map = {name: i for i, name in enumerate(class_names)}
    
    # convert labels to one-hot encoding
    train_labels = labels_to_onehot(train_labels, label_map)
    test_labels = labels_to_onehot(test_labels, label_map)
    
    return train_data, train_labels, test_data, test_labels, label_map 
