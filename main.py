import os
from sklearn.utils.class_weight import compute_class_weight
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.models import load_model
import argparse
import numpy as np
from lenet import LeNet
from mobilenet import MobileNet
from data_grocery import get_data, prepare_data, get_fruits360, prepare_fruits360

PRETRAINED_PATH = "fruits360_pretrain.keras"


# trains on Fruits-360 so the model learns general produce features (color, shape, texture)
# before it ever sees the grocery store photos
def pretrain(img_rows, img_cols, channels, batch_size):
    image_dir = get_fruits360('./fruits360_kaggle')
    train_ds, test_ds, num_classes = prepare_fruits360(image_dir, img_rows, img_cols, batch_size)
    print(f"Pretraining on Fruits-360 with {num_classes} classes...")

    model = LeNet.build(channels, img_rows, img_cols, num_classes)
    model.compile(optimizer=SGD(learning_rate=0.01, momentum=0.9), loss="categorical_crossentropy", metrics=["accuracy"])
    early_stop = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)
    model.fit(train_ds, epochs=10, validation_data=test_ds, callbacks=[early_stop], verbose=1)

    print(f"Saving pretrained model to {PRETRAINED_PATH}...")
    model.save(PRETRAINED_PATH)


# trains LeNet on the grocery data, starting from the Fruits-360 pretrained layers if they exist
def train_lenet(args, train, val, img_rows, img_cols, channels, batch_size, weights_path):
    num_classes = train[1].shape[1]
    print("Building LeNet model...")
    model = LeNet.build(channels, img_rows, img_cols, num_classes)

    # if load model parser is added then use the saved grocery weights,
    # otherwise start from the Fruits-360 pretrained layers if they exist
    if args["load_model"] == 1:
        print("Loading pre-trained weights...")
        model.load_weights(weights_path)
    elif os.path.exists(PRETRAINED_PATH) and args["scratch"] != 1:
        print(f"Starting from Fruits-360 pretrained layers in {PRETRAINED_PATH}...")
        LeNet.transfer_weights(load_model(PRETRAINED_PATH), model)

    model.compile(optimizer=SGD(learning_rate=0.01, momentum=0.9), loss="categorical_crossentropy", metrics=["accuracy"])

    # stop once validation loss hasn't improved for 8 epochs and keep the best weights seen
    early_stop = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)

    print("Training the model...")
    model.fit(train[0], train[1], batch_size=batch_size, epochs=50,
              validation_data=val, callbacks=[early_stop], verbose=1)
    return model


# trains MobileNetV2 in two phases:
# 1. the ImageNet layers stay frozen and only the new classifier learns
# 2. the top of the ImageNet layers is unfrozen and fine-tuned with a much smaller learning rate
def train_mobilenet(args, train, val, img_rows, img_cols, channels, batch_size, weights_path):
    num_classes = train[1].shape[1]
    print("Building MobileNetV2 model...")
    model, base = MobileNet.build(channels, img_rows, img_cols, num_classes)

    if args["load_model"] == 1:
        print("Loading pre-trained weights...")
        model.load_weights(weights_path)

    print("Phase 1: training the classifier on top of frozen ImageNet layers...")
    model.compile(optimizer=Adam(learning_rate=1e-3), loss="categorical_crossentropy", metrics=["accuracy"])
    early_stop = EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)
    model.fit(train[0], train[1], batch_size=batch_size, epochs=15,
              validation_data=val, callbacks=[early_stop], verbose=1)

    print("Phase 2: fine-tuning the top ImageNet layers...")
    MobileNet.unfreeze(base, 80)
    model.compile(optimizer=Adam(learning_rate=1e-5), loss="categorical_crossentropy", metrics=["accuracy"])
    early_stop = EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, min_lr=1e-7, verbose=1)
    train_labels = np.argmax(train[1], axis=1)
    class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(train_labels), y=train_labels)
    class_weights = dict(enumerate(class_weights))
    model.fit(train[0], train[1], batch_size=batch_size, epochs=40,
              validation_data=val, callbacks=[early_stop, reduce_lr], 
              class_weight=class_weights, verbose=1)
    return model


def main():
    # parameters for training
    # batch size is how many images are processed per training step
    # img rows, columns and channels set preprocessing parameters to be read by the model
    # (color matters a lot for telling produce apart, so images are loaded as RGB)
    batch_size = 32
    channels = 3

    # add argument parsers to model
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", choices=["lenet", "mobilenet"], default="lenet",
                    help="lenet trains from scratch, mobilenet starts from a model pretrained on ImageNet")
    ap.add_argument("-l", "--load-model", type=int, default=-1)
    ap.add_argument("-c", "--coarse", type=int, default=0,
                    help="1 to train on the 42 coarse classes instead of the 81 fine classes")
    ap.add_argument("-p", "--pretrain", type=int, default=0,
                    help="lenet only: 1 to pretrain on Fruits-360 (downloads an 8GB zip) before training on the grocery data")
    ap.add_argument("-s", "--scratch", type=int, default=0,
                    help="lenet only: 1 to ignore the Fruits-360 pretrained model and train from scratch")
    args = vars(ap.parse_args())
    coarse = args["coarse"] == 1
    use_mobilenet = args["model"] == "mobilenet"

    # MobileNetV2's ImageNet weights come in fixed sizes; 160 balances accuracy against memory and speed
    img_rows = img_cols = 192 if use_mobilenet else 64
    prefix = "grocery_mobilenet" if use_mobilenet else "grocery"
    weights_path = f"{prefix}_coarse.weights.h5" if coarse else f"{prefix}_fine.weights.h5"

    if args["pretrain"] == 1 and not use_mobilenet:
        pretrain(img_rows, img_cols, channels, batch_size)

    # retrieves data from dataset path, if it doesn't exist then downloads and unzips from Kaggle
    download_path = './grocery_kaggle'
    get_data(download_path)
    dataset_path = './grocery_kaggle/GroceryStoreDataset'

    train, val, (test_data, test_labels), label_map = \
        prepare_data(dataset_path, img_rows, img_cols, channels, coarse)
    print(f"{len(train[0])} train / {len(val[0])} val / {len(test_data)} test images, {train[1].shape[1]} classes")

    trainer = train_mobilenet if use_mobilenet else train_lenet
    model = trainer(args, train, val, img_rows, img_cols, channels, batch_size, weights_path)

    # evaluate the model
    print("Evaluating the model...")
    loss, accuracy = model.evaluate(test_data, test_labels, batch_size=batch_size)
    print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

    # save model weights
    print("Saving model weights...")
    model.save_weights(weights_path)

if __name__ == '__main__':
    main()
