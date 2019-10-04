#!/usr/bin/env python
# -*- coding: utf-8 -*-

from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, Concatenate
from keras.models import Model
from keras.optimizers import Adam

from settings import target_coordinates

__all__ = ["get_model", "masked_mse"]


def masked_mse(target, prediction, unknown_label=-1.0):
    """
    The masked mean-squared error (MSE) loss function as described in section IV.E of the paper "Learning Long-Range
    Perception Using Self-Supervision from Short-Range Sensors and Odometry" (https://arxiv.org/abs/1809.07207).

    :param target: a tensor that represents the ground-truth labels for target.

    :param prediction: the model prediction of the target's label.

    :param unknown_label: a value that represents the unknown label.

    :return: the masked MSE loss between target and prediction.
    """
    # mask is a tensor of the same shape as target with 1 where target is not the unknown_label and 0 otherwise.
    mask = K.cast(K.not_equal(target, unknown_label), K.floatx())

    # Only known labels contribute to the loss.
    mse = K.mean(K.square((prediction - target) * mask))

    return mse


def get_model(features=["camera"], targets=["target"], learning_rate=0.001, show_summary=False):
    """
    Creates a CNN model given the inputs (features) and the corresponding outputs (targets).

    :param features: a list of strings that contains the name of each input (the name of each long-range sensor).

    :param targets: a list of strings that contains the name of each output (the name of each short-range sensor)

    :param learning_rate: the learning rate of the optimizer (used during training).

    :param show_summary: If true, info about the CNN is shown to the console.

    :return: an object of class Model.
    """
    if len(set(features)) != len(features):
        raise ValueError("features contains duplicates")
    if len(set(targets)) != len(targets):
        raise ValueError("targets contains duplicates")

    # camera should also correspond to the key in the HDF5 file.
    inputs = [Input(shape=(64, 80, 3), name=feature) for feature in features]

    def convolution_2d(inp, filters):
        result = Conv2D(filters, (3, 3), padding='same', activation='relu')(inp)
        result = MaxPooling2D(pool_size=(2, 2))(result)
        return result

    convolution_inputs = Concatenate(axis=-1)(inputs) if len(inputs) > 1 else inputs[0]
    c = convolution_2d(convolution_inputs, 20)
    c = convolution_2d(c, 12)
    c = convolution_2d(c, 10)
    c = convolution_2d(c, 8)

    fc = Flatten()(c)
    fc = Dense(512, activation='relu')(fc)

    outputs = []
    for target in targets:
        outputs.append(Dense(len(target_coordinates), activation='sigmoid', name=target)(fc))

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(loss={target: masked_mse for target in targets}, optimizer=Adam(lr=learning_rate))

    if show_summary:
        model.summary()

    return model
