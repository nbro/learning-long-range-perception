#!/usr/bin/python

"""Function to create the keras model and prepare the data before the training or testing phase."""

import numpy as np
from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from keras.models import Model
from keras.optimizers import Adam

from settings import coords


def flip(x, y):
    """Flips an image and the corresponding labels.

    Args:
        x: an image represented by a 3d numpy array
        y: a list of labels associated with the image

    Returns:
        the flipped image and labels.
    """
    if np.random.choice([True, False]):
        x = np.fliplr(x)

        for i in range(len(y) // 5):
            y[i * 5:(i + 1) * 5] = np.flipud(y[i * 5:(i + 1) * 5])

    return (x, y)


def model(lr=0.001, show_summary=False):
    """Creates the keras neural network model.

    Args:
        lr: the learning rate used for the training.
        show_summary: a boolean flag that represents if the model has to be printed to console.

    Returns:
        The defined keras model.
    """
    input_cam1 = Input(shape=(64, 80, 3), name='input_cam1')

    def conv2d(inp, filters):
        result = Conv2D(filters, (3, 3), padding='same', activation='relu')(inp)
        result = MaxPooling2D(pool_size=(2, 2))(result)
        return result

    conv_part = conv2d(input_cam1, 20)
    conv_part = conv2d(conv_part, 12)
    conv_part = conv2d(conv_part, 10)
    conv_part = conv2d(conv_part, 8)

    ff_part = Flatten()(conv_part)
    ff_part = Dense(512, activation='relu')(ff_part)

    outputs = []
    target_columns = ['target1']
    for label in target_columns:
        outputs.append(Dense(len(coords), activation='sigmoid', name='output_' + label)(ff_part))

    model = Model(inputs=input_cam1, outputs=outputs)

    def masked_mse(target, pred):
        mask = K.cast(K.not_equal(target, -1), K.floatx())
        mse = K.mean(K.square((pred - target) * mask))
        return mse

    model.compile(loss={'output_' + label: masked_mse for label in target_columns}, optimizer=Adam(lr=lr))

    if show_summary:
        model.summary()

    return model
