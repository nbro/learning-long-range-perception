#!/usr/bin/env python
# -*- coding: utf-8 -*-

import h5py
import numpy as np


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

    return x, y


def make_random_gradient(size):
    """Creates a random gradient

    Args:
        size: the size of the gradient

    Returns:
        the random gradient.
    """
    x, y = np.meshgrid(np.linspace(0, 1, size[1]), np.linspace(0, 1, size[0]))
    grad = x * np.random.uniform(-1, 1) + y * np.random.uniform(-1, 1)
    grad = (grad - grad.mean()) / grad.std()
    return grad


# TODO: understand what this function does and why we use it.
def apply_random_gradient(x):
    """Applies a random gradient to the image

    Args:
        x: an image represented by a 3d numpy array

    Returns:
        the image with the added random gradient.
    """
    grad = make_random_gradient(x.shape)

    for i in range(3):
        x[:, :, i] = x[:, :, i] * np.random.uniform(0.9, 1.1)
    x = (x - x.mean()) / x.std()

    amount = np.random.uniform(0.05, 0.15)

    for i in range(3):
        x[:, :, i] = x[:, :, i] * (1 - amount) + grad * amount
    x = (x - x.mean()) / x.std()

    return x


def add_gaussian_noise(x):
    """Adds gaussian noise centered on 0 to an image.

    Args:
        x: an image represented by a 3d numpy array

    Returns:
        the image with the added noise.
    """
    gauss = np.random.normal(0, 2 * 1e-2, x.shape)  # 2% gaussian noise
    x = x + gauss
    return x


def to_grayscale(x):
    """Converts an image to to_grayscale.

    Args:
        x: an image represented by a 3d numpy array

    Returns:
        the to_grayscale image.
    """
    return np.dstack([0.21 * x[:, :, 2] + 0.72 * x[:, :, 1] + 0.07 * x[:, :, 0]] * 3)


def random_augment(im):
    """See section 4.D of the paper "Learning Long-Range Perception Using Self-Supervision from Short-Range Sensors and
    Odometry"

    :param im: the image to augment

    :return: a modified version of im.
    """
    # With probability 1/3, either add Gaussian noise, convert the image to grayscale or do nothing. However, it always
    # applies a random gradient.
    choice = np.random.randint(0, 3)

    if choice == 0:
        im = add_gaussian_noise(im)
    elif choice == 1:
        im = to_grayscale(im)

    im = (im - im.mean()) / im.std()
    im = apply_random_gradient(im)

    return im


def get_generator(split_percentage=50.0, hdf5_file_name=None, batch_size=1, augment=True, is_testset=False,
                  testset_index=0, unknown_label=-1.0):
    """Loads the dataset, preprocess it and generates batches of data.

    Args:
        split_percentage: a percentage (from 0 to 100) representing the split between training and testing sets.
        hdf5_file_name: a filename for an hdf5 storage.
        batch_size: the size of the batch.
        augment: a boolean flag representing wether to augment the data or not.
        is_testset: a boolean flag representing wether to generate data for the testing or training.
        testset_index: the index of the hdf5 storage data to be used as testset.

    Returns:
        the preprocessed batches.
    """
    if hdf5_file_name is None:
        raise ValueError("hdf5_file_name cannot be None")

    hdf5_file = h5py.File(hdf5_file_name, 'r')

    n_bags = len(hdf5_file.keys())

    # Create an array of integers, one for each of the bag files.
    bag_indices = np.arange(0, n_bags)

    # Get the name of the features (long-range sensors) and targets (short-range sensors).
    features = [target for target in hdf5_file['bag0/features'].keys() if target not in ['x', 'y', 'angle']]
    targets = [target for target in hdf5_file['bag0/targets'].keys()]

    # Create a dictionary from indices (one index associated with each bag file in the HDF5 file) to dictionaries (one
    # dictionary associated with each bag file), which are maps from the features (or targets) to the corresponding
    # values of those features (or targets).
    Xs = {i: {feature: hdf5_file['bag' + str(i) + '/features/' + feature] for feature in features} for i in bag_indices}
    Ys = {i: {target: hdf5_file['bag' + str(i) + '/targets/' + target] for target in targets} for i in bag_indices}

    lengths = {i: Xs[i][features[0]].shape[0] for i in bag_indices}

    counts = {i: 0 for i in bag_indices}

    def binarise_outputs(class_1=1.0, class_2=0.0):
        for target in targets:
            out = outputs[target]
            out[(0 <= out) & (out <= 128)] = class_1
            out[out > 128] = class_2

    if is_testset:
        # Chose the index of the bag that will be used as test dataset.
        index = n_bags - 1 if testset_index == -1 else testset_index

        inputs = {feature: Xs[index][feature][:] for feature in features}
        outputs = {target: Ys[index][target][:] for target in targets}

        masks = {'mask_' + target: Ys[index][target][:] != unknown_label for target in targets}

        binarise_outputs()

        yield (inputs, outputs, masks)

    else:
        group = bag_indices[:int(np.ceil(n_bags * split_percentage / 100.0))]

        while True:
            # Choose a bag file from the training bag files.
            index = np.random.choice(group)

            inputs = {feature: Xs[index][feature][counts[index]:counts[index] + batch_size] for feature in features}
            outputs = {target: Ys[index][target][counts[index]:counts[index] + batch_size] for target in targets}

            counts[index] += batch_size
            if counts[index] + batch_size > lengths[index]:
                counts[index] = 0

            # In section 4.D of the paper "Learning Long-Range Perception Using Self-Supervision from Short-Range
            # Sensors and Odometry", the authors state that they augment the data, so that to increase the size of the
            # dataset. For example, with probability 0.5 they flip the image and the corresponding target labels are
            # modified by swapping the outputs of the left and right sensors and the outputs of the center-left and
            # center-right sensors. Of course, this is valid in the case of the Thymio robot, but not the Pioneer3AT
            # one.
            if augment:
                for feature in features:
                    inp = inputs[feature]
                    for i in range(batch_size):
                        inp[i] = random_augment(inp[i])

            binarise_outputs()

            yield (inputs, outputs)
