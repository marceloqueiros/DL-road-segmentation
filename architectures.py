from __future__ import print_function

import os
import sys
import logging

import numpy as np
from keras import backend as K

from keras.callbacks import ModelCheckpoint
from keras.layers import (Activation, Conv2D, Conv2DTranspose, Dense, Dropout,
                          Flatten, Input, MaxPooling2D, concatenate,
                          GlobalAveragePooling2D)
from keras.models import Model, Sequential
from keras.optimizers import Adam, RMSprop

#from .SqueezeNet import SqueezeNet
from keras.applications.resnet50 import ResNet50
from keras.applications.mobilenet import MobileNet
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.densenet import DenseNet, DenseNet121, DenseNet169, DenseNet201
from keras.applications.nasnet import NASNet, NASNetLarge, NASNetMobile

import keras.metrics

#import keras_resnet.models

K.set_image_data_format("channels_last")

architectures_log = logging.getLogger("architectures")


# *****************************************************************************
# Custom metrics
# *****************************************************************************
def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)

    return (2. * intersection + smooth) / (
        K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)


def custom_acc(threshold=0.75):
    def acc_conf(y_true, y_pred):
        # Get correct predictions
        matches = K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1))

        # Get predictions with confidence higher than 75%
        conf = K.greater_equal(K.max(y_pred, axis=-1), threshold)

        # Get correct predicitons with confidence higher than 75%
        res = matches & conf

        return K.mean(res)

    return acc_conf


def custom_loss(y_true, y_pred):
    # Get custom metric
    acc = custom_acc(threshold=0.75)

    return K.mean(K.sign(acc) * acc)


# *****************************************************************************
# Custom classification network architectures
# *****************************************************************************
def cnn(input_tensor, input_shape, nb_classes):
    architectures_log.info("Architecture: CustomCNN")

    conv1 = Conv2D(8, (3, 3), strides=(2, 2), activation="relu", padding="same")(input_tensor)
    conv1 = Conv2D(8, (3, 3), strides=(2, 2), activation="relu", padding="same")(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    """
    conv2 = Conv2D(16, (3, 3), activation="relu", padding="same")(pool1)
    conv2 = Conv2D(16, (3, 3), activation="relu", padding="same")(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(32, (3, 3), activation="relu", padding="same")(pool2)
    conv3 = Conv2D(32, (3, 3), activation="relu", padding="same")(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    """

    flat = Flatten()(pool1)

    dense1 = Dense(128, activation="relu")(flat)
    drop1 = Dropout(0.5)(dense1)
    
    """
    dense2 = Dense(512, activation="relu")(drop1)
    drop2 = Dropout(0.5)(dense2)
    """
    
    output_tensor = Dense(nb_classes, activation="softmax", name="output")(drop1)

    # initiate RMSprop optimizer
    opt = Adam(lr=0.001, decay=1e-6)

    model = Model(inputs=[input_tensor], outputs=[output_tensor])

    # Let's train the model using RMSprop
    model.compile(
        loss="categorical_crossentropy",
        optimizer=opt,
        metrics=["accuracy"])
        #metrics=["accuracy", custom_acc(0.75)])

    print(model.summary())
    architectures_log.info("Architecture: CustomCNN")

    return model


# Custom convolutional block
def custom_conv_block(input_tensor,
                      n_filters=8,
                      f_size=(3, 3),
                      strides=(1, 1),
                      padding="same",
                      activation="relu"):

    block = None

    return block


# Dynamic module builder
def build_dynamic(input_tensor,
                  input_shape,
                  nb_classes,
                  depth=1,
                  width=1,
                  n_filters=8,
                  f_size=(3, 3),
                  activation="relu",
                  paddig="same",
                  fc_units=512,
                  dropout=0.5):

    network = custom_conv_block(input_tensor=input_tensor, n_filters=n_filters)

    # Controls number of blocks
    for d in range(depth):

        # Controls number of layers per block
        for w in range(width):

            network = custom_conv_block(
                input_tensor=network, n_filters=n_filters * d)

        network = MaxPooling2D(pool_size=(2, 2))(network)

    network = Flatten()(network)

    for w in width:
        network = Dense(fc_units // (w + 1), activation="relu")(network)

        if (dropout > 0. and dropout <= 1.):
            network = Dropout(dropout)(network)

    network = Dense(units=nb_classes, activation="softmax")(network)

    model = Model(inputs=[input_tensor], outputs=[network])

    return model


# *****************************************************************************
# Custom segmentation network architectures
# *****************************************************************************
def unet(input_tensor, input_shape=None, nb_classes=None):
    # Makes 4 downsamplings
    # 2^4=16 » Minimum image size allowed is 32x32
    architectures_log.info("Architecture: U-Net")
    conv1 = Conv2D(32, (3, 3), activation="relu", padding="same")(input_tensor)
    conv1 = Conv2D(32, (3, 3), activation="relu", padding="same")(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation="relu", padding="same")(pool1)
    conv2 = Conv2D(64, (3, 3), activation="relu", padding="same")(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation="relu", padding="same")(pool2)
    conv3 = Conv2D(128, (3, 3), activation="relu", padding="same")(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation="relu", padding="same")(pool3)
    conv4 = Conv2D(256, (3, 3), activation="relu", padding="same")(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation="relu", padding="same")(pool4)
    conv5 = Conv2D(512, (3, 3), activation="relu", padding="same")(conv5)

    up6 = concatenate(
        [
            Conv2DTranspose(256, (2, 2), strides=(2, 2),
                            padding="same")(conv5), conv4
        ],
        axis=3)
    conv6 = Conv2D(256, (3, 3), activation="relu", padding="same")(up6)
    conv6 = Conv2D(256, (3, 3), activation="relu", padding="same")(conv6)

    up7 = concatenate(
        [
            Conv2DTranspose(128, (2, 2), strides=(2, 2),
                            padding="same")(conv6), conv3
        ],
        axis=3)
    conv7 = Conv2D(128, (3, 3), activation="relu", padding="same")(up7)
    conv7 = Conv2D(128, (3, 3), activation="relu", padding="same")(conv7)

    up8 = concatenate(
        [
            Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(conv7),
            conv2
        ],
        axis=3)
    conv8 = Conv2D(64, (3, 3), activation="relu", padding="same")(up8)
    conv8 = Conv2D(64, (3, 3), activation="relu", padding="same")(conv8)

    up9 = concatenate(
        [
            Conv2DTranspose(32, (2, 2), strides=(2, 2), padding="same")(conv8),
            conv1
        ],
        axis=3)
    conv9 = Conv2D(32, (3, 3), activation="relu", padding="same")(up9)
    conv9 = Conv2D(32, (3, 3), activation="relu", padding="same")(conv9)

    conv10 = Conv2D(1, (1, 1), activation="sigmoid")(conv9)

    model = Model(inputs=[input_tensor], outputs=[conv10])

    model.compile(
        optimizer=Adam(lr=1e-4),
        loss=["mean_squared_error"],
        metrics=["mean_squared_error"])

    print(model.summary())
    architectures_log.info("Architecture: U-Net")

    return model


# *****************************************************************************
# Keras applications network architectures
# https://keras.io/applications/#documentation-for-individual-models
# *****************************************************************************
def squeezenet(input_tensor, input_shape, nb_classes):
    architectures_log.info("Architecture: SqueezeNet")

    model = SqueezeNet(
        include_top=True,
        weights=None,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling="avg",
        classes=nb_classes
    )

    opt = Adam(lr=0.0001, decay=1e-6)

    # Let"s train the model using RMSprop
    model.compile(
        loss="categorical_crossentropy",
        optimizer=opt,
        metrics=["accuracy", custom_acc(0.75)])

    print(model.summary())
    architectures_log.info("Architecture: SqueezeNet")

    return model


def mobilenet(input_tensor, input_shape, nb_classes):
    architectures_log.info("Architecture: MobileNet")

    model = MobileNet(
        include_top=True,
        weights=None,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling="avg",
        classes=nb_classes
    )

    opt = Adam(lr=0.0001, decay=1e-6)

    model.compile(
        loss="categorical_crossentropy",
        optimizer=opt,
        metrics=["accuracy", custom_acc(0.75)])

    print(model.summary())
    architectures_log.info("Architecture: MobileNet")

    return model


# 
def build_resnetN(input_tensor, input_shape, nb_classes):
    """
    Pre-built N-layer Residual Neural Network.
    N = {18, 34, 50, 101, 152, 200}
    
    Arguments:
        input_tensor {tensor} -- Input tensor.
        input_shape {tuple} -- Shape of input image.
        nb_classes {int} -- Number of classes.
    
    Returns:
        Model -- N-layer ResNet model.
    """
    architectures_log.info("Architecture: ResNet18")

    model = keras_resnet.models.ResNet34(
        inputs=input_tensor, blocks=None, include_top=True, classes=nb_classes)

    opt = Adam(lr=0.001, decay=1e-6)

    model.compile(
        loss="categorical_crossentropy",
        optimizer=opt,
        metrics=["accuracy"])

    print(model.summary())
    architectures_log.info("Architecture: ResNet18")

    return model


def resnet50(input_tensor, input_shape, nb_classes):
    """
    Pre-built 50-layer Residual Neural Network.
    
    Arguments:
        input_tensor {tensor} -- Input tensor.
        input_shape {tuple} -- Shape of input image.
        nb_classes {int} -- Number of classes.
    
    Returns:
        Model -- 50-layer ResNet model.
    """

    architectures_log.info("Architecture: ResNet50")

    model = ResNet50(
        include_top=True,
        weights=None,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling="avg",
        classes=nb_classes)

    opt = Adam(lr=0.001, decay=1e-6)

    model.compile(
        loss="categorical_crossentropy",
        optimizer=opt,
        metrics=["accuracy", custom_acc(threshold=0.75)])

    print(model.summary())
    architectures_log.info("Architecture: ResNet50")

    return model


def densenet121(input_tensor, input_shape, nb_classes):
    """
    Pre-built DenseNet 121-layer Residual Neural Network.
    
    Arguments:
        input_tensor {tensor} -- Input tensor.
        input_shape {tuple} -- Shape of input image.
        nb_classes {int} -- Number of classes.
    
    Returns:
        Model -- 121-layer DenseNet model.
    """

    architectures_log.info("Architecture: DenseNet121")

    model = DenseNet121(
        include_top=True,
        weights=None,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling="avg",
        classes=nb_classes)

    opt = Adam(lr=0.0001, decay=1e-6)

    model.compile(
        loss="categorical_crossentropy",
        optimizer=opt,
        metrics=["accuracy", custom_acc(threshold=0.75)])

    print(model.summary())
    architectures_log.info("Architecture: DenseNet121")

    return model


def densenet201(input_tensor, input_shape, nb_classes):
    """
    Pre-built DenseNet 201-layer Residual Neural Network.
    
    Arguments:
        input_tensor {tensor} -- Input tensor.
        input_shape {tuple} -- Shape of input image.
        nb_classes {int} -- Number of classes.
    
    Returns:
        Model -- 201-layer DenseNet model.
    """

    architectures_log.info("Architecture: DenseNet201")

    model = DenseNet201(
        include_top=True,
        weights=None,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling="avg",
        classes=nb_classes)

    opt = Adam(lr=0.0001, decay=1e-6)

    model.compile(
        loss="categorical_crossentropy",
        optimizer=opt,
        metrics=["accuracy", custom_acc(threshold=0.75)])

    print(model.summary())
    architectures_log.info("Architecture: DenseNet201")

    return model


def inception_resnetv2(input_tensor, input_shape, nb_classes):
    architectures_log.info("Architecture: InceptionResnetV2")

    model = InceptionResNetV2(
        include_top=True,
        weights=None,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling="max",
        classes=nb_classes)

    opt = Adam(lr=0.001, decay=1e-6)

    model.compile(
        loss="categorical_crossentropy",
        optimizer=opt,
        metrics=["accuracy", custom_acc(0.75)])

    print(model.summary())
    architectures_log.info("Architecture: InceptionResnetV2")

    return model
