#!/usr/bin/python

"""Function to create the keras model and prepare the data before the training or testing phase."""

from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from keras.models import Model
from keras.optimizers import Adam

from settings import target_coordinates


def model(lr=0.001, show_summary=False):
    """Creates the keras neural network model.

    Args:
        lr: the learning rate used for the training.
        show_summary: a boolean flag that represents if the model has to be printed to console.

    Returns:
        The defined keras model.
    """
    # camera should also correspond to the key in the HDF5 file.
    input_camera = Input(shape=(64, 80, 3), name='input_camera')

    def convolution_2d(inp, filters):
        result = Conv2D(filters, (3, 3), padding='same', activation='relu')(inp)
        result = MaxPooling2D(pool_size=(2, 2))(result)
        return result

    c = convolution_2d(input_camera, 20)
    c = convolution_2d(c, 12)
    c = convolution_2d(c, 10)
    c = convolution_2d(c, 8)

    fc = Flatten()(c)
    fc = Dense(512, activation='relu')(fc)

    outputs = []
    targets = ['target']  # "target" should also correspond to the key in the HDF5 file.

    for target in targets:
        outputs.append(Dense(len(target_coordinates), activation='sigmoid', name='output_' + target)(fc))

    model = Model(inputs=input_camera, outputs=outputs)

    # TODO: understand
    def masked_mse(target, pred):
        mask = K.cast(K.not_equal(target, -1), K.floatx())
        mse = K.mean(K.square((pred - target) * mask))
        return mse

    model.compile(loss={'output_' + target: masked_mse for target in targets}, optimizer=Adam(lr=lr))

    if show_summary:
        model.summary()

    return model
