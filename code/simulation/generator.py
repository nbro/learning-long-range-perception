#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
from os import path

import h5py
import numpy as np


def normalize(x):
    """It subtracts the mean of x from x, then divides the result by the standard deviation of x.

    :param x: a 3d numpy array (which e.g. represents an image).

    :return: the normalised version of x.
    """
    return (x - x.mean()) / x.std()


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


# TODO: understand this function
def make_random_gradient(size):
    x, y = np.meshgrid(np.linspace(0, 1, size[1]), np.linspace(0, 1, size[0]))
    grad = x * np.random.uniform(-1, 1) + y * np.random.uniform(-1, 1)
    grad = normalize(grad)
    return grad


# TODO: understand this function
def apply_random_gradient(x, amount=np.random.uniform(0.05, 0.15), perturb_channels=True):
    grad = make_random_gradient(x.shape)

    if perturb_channels:
        for i in range(3):
            x[:, :, i] = x[:, :, i] * np.random.uniform(0.9, 1.1)
        x = normalize(x)

    for i in range(3):
        x[:, :, i] = (1 - amount) * x[:, :, i] + amount * grad

    x = normalize(x)

    return x


def add_gaussian_noise(x, mu=0., sigma=0.5):
    gauss = np.random.normal(mu, sigma, x.shape)  # 2% gaussian noise
    x = x + gauss
    return x


def to_grayscale(x):
    """Converts an image to grayscale.

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


def augment_inputs(inputs):
    """
    See section 4.D of the paper "Learning Long-Range Perception Using Self-Supervision from Short-Range Sensors and
    Odometry".

    :param inputs: a dictionary from names of features to the corresponding batches of data.
    """
    for feature, batch in inputs.iteritems():
        for i in range(len(batch)):
            batch[i] = random_augment(batch[i])


def binarize_outputs(outputs, class_1=1.0, class_2=0.0, rgb_threshold=128):
    for target, batch in outputs.iteritems():
        batch[(0 <= batch) & (batch <= rgb_threshold)] = class_1
        batch[batch > rgb_threshold] = class_2


def get_generator(hdf5_file_name=None, bag_id="bag0", features_group="features", targets_group="targets",
                  features=["camera"], targets=["target"], batch_size=1, split_percentage=50.0, augment=True,
                  is_testset=False, unknown_label=-1.0):
    if hdf5_file_name is None:
        raise ValueError("hdf5_file_name cannot be None.")

    hdf5_file = h5py.File(hdf5_file_name, "r")

    if len(hdf5_file.keys()) == 0:
        raise ValueError("HDF5 file with name {} is empty.".format(hdf5_file_name))

    if bag_id is None:
        bag_id = random.choice(hdf5_file.keys())

    if bag_id not in hdf5_file.keys():
        raise ValueError("{} is not a key in the dataset {}.".format(bag_id, hdf5_file_name))

    for feature in features:
        if not path.join(bag_id, features_group, feature) in hdf5_file:
            raise ValueError("feature {} does not exist in the dataset.".format(feature))

    for target in targets:
        if not path.join(bag_id, targets_group, target) in hdf5_file:
            raise ValueError("target {} does not exist in the dataset.".format(target))

    # The keys to index the features and the targets in the HDF5 file (dictionary).
    features_keys = [path.join(bag_id, features_group, feature) for feature in features]
    targets_keys = [path.join(bag_id, targets_group, target) for target in targets]

    n_observations = hdf5_file[features_keys[0]].shape[0]

    for key in features_keys + targets_keys:
        if hdf5_file[key].shape[0] != n_observations:
            raise ValueError("not all features and targets have the same number of observations.")

    if split_percentage < 1.0 or split_percentage > 99.0:
        raise ValueError("split_percentage must be in the range [1.0, 99.0].")

    counter = 0

    # All observations from 0 to n_training_examples are used for training. All other observations are used for testing.
    n_training_examples = int(np.ceil(n_observations * split_percentage / 100.0))

    if is_testset:
        # The validation set.
        inputs = {feature: hdf5_file[key][n_training_examples:] for feature, key in zip(features, features_keys)}
        outputs = {target: hdf5_file[key][n_training_examples:] for target, key in zip(targets, targets_keys)}

        # Test the model (calculate the AUC) only with the inputs that have an associated label.
        masks = {target: hdf5_file[key][n_training_examples:] != unknown_label for target, key in
                 zip(targets, targets_keys)}

        binarize_outputs(outputs)

        yield (inputs, outputs, masks)

    else:
        while True:
            # The start and end indices of the batch.
            start = counter
            end = min(counter + batch_size, n_training_examples)

            # The batch of inputs and outputs.
            inputs = {feature: hdf5_file[key][start:end] for feature, key in zip(features, features_keys)}
            outputs = {target: hdf5_file[key][start:end] for target, key in zip(targets, targets_keys)}

            # Update the counter, so that to get a new batch at the next iteration.
            if counter + batch_size >= n_training_examples:
                counter = 0
            else:
                counter += batch_size

            if augment:
                augment_inputs(inputs)

            binarize_outputs(outputs)

            yield (inputs, outputs)
