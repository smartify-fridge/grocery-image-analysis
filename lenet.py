from keras.models import Sequential
from keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Activation,
                          GlobalAveragePooling2D, Dropout, Input, RandomFlip, RandomRotation,
                          RandomZoom, RandomContrast)
from keras import models

# parameters and layers based off of LeNet model
class LeNet(models.Sequential):
    @staticmethod
    def build(channels, im_rows, im_cols, nb_classes, weights_path=None):
        input_shape = (im_rows, im_cols, channels)

        model = Sequential()
        model.add(Input(shape=input_shape))

        # data augmentation, only active during training
        # gives the model varied versions of each image so it can't just memorize them
        model.add(RandomFlip("horizontal"))
        model.add(RandomRotation(0.1))
        model.add(RandomZoom(0.15))
        model.add(RandomContrast(0.2))

        # #original LeNet layers
        # model.add(Conv2D(filters=32, kernel_size=(5,5), padding='same', activation='relu', input_shape = input_shape))
        # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # model.add(Conv2D(filters=48, kernel_size=(5,5), padding='valid', activation='relu'))
        # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # model.add(Conv2D(120, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'))
        # model.add(Flatten())
        # model.add(Dense(256, activation='relu'))
        # model.add(Dense(nb_classes, activation='softmax'))

        # LeNet layers with normalization, relu activations and dropout
        model.add(Conv2D(32, kernel_size=(5, 5), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
        model.add(Conv2D(64, kernel_size=(5, 5), padding='valid'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
        model.add(Conv2D(128, kernel_size=(3, 3), padding='valid'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        # averaging each feature map instead of flattening cuts the dense layer from ~9.4M weights to ~66K
        model.add(GlobalAveragePooling2D())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(nb_classes, activation='softmax'))

        # checks to see if there are saved weight paths
        if weights_path is not None:
            model.load_weights(weights_path)

        return model

    # copies every layer except the final classifier from a pretrained model,
    # so a model trained on one set of classes can start training on another
    @staticmethod
    def transfer_weights(pretrained, model):
        for old_layer, new_layer in zip(pretrained.layers[:-1], model.layers[:-1]):
            new_layer.set_weights(old_layer.get_weights())
