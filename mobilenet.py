from keras import Model
from keras.applications import MobileNetV2
from keras.layers import (Input, Dense, Dropout, GlobalAveragePooling2D, Rescaling,
                          RandomFlip, RandomRotation, RandomZoom, RandomContrast, RandomTranslation)

# MobileNetV2 pretrained on ImageNet (1.4M images, 1000 classes) with a new grocery classifier on top
class MobileNet:
    @staticmethod
    def build(channels, im_rows, im_cols, nb_classes):
        inputs = Input(shape=(im_rows, im_cols, channels))

        # data augmentation, only active during training
        x = RandomFlip("horizontal")(inputs)
        x = RandomRotation(0.1)(x)
        x = RandomZoom(0.15)(x)
        x = RandomTranslation(0.1, 0.1)(x)
        x = RandomContrast(0.2)(x)

        # images come in as [0, 1] but MobileNetV2 was trained on [-1, 1]
        x = Rescaling(2.0, offset=-1.0)(x)

        # downloads the ImageNet weights (~9MB) on first use; include_top=False drops its 1000-class classifier
        base = MobileNetV2(input_shape=(im_rows, im_cols, channels), include_top=False, weights="imagenet")
        base.trainable = False
        # training=False keeps the base's batch norm statistics from ImageNet, even while fine-tuning
        x = base(x, training=False)

        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation="relu")(x)
        x = Dropout(0.4)(x)
        outputs = Dense(nb_classes, activation="softmax")(x)

        return Model(inputs, outputs), base

    # lets the last n layers of the pretrained base keep learning, leaving batch norm layers frozen
    @staticmethod
    def unfreeze(base, n_layers):
        base.trainable = True
        for layer in base.layers[:-n_layers]:
            layer.trainable = False
        for layer in base.layers[-n_layers:]:
            if layer.__class__.__name__ == "BatchNormalization":
                layer.trainable = False
