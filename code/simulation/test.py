#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import time
from os import path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tqdm
from sklearn.metrics import roc_auc_score

from generator import get_generator
from model import get_model
from settings import target_coordinates
from utils import percent

DEFAULT_MODEL = "2019-10-12-15-46-25/10"


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('-w', '--weights-folder', type=str, default="models/{}/weights".format(DEFAULT_MODEL),
                        help='The name of the folder that contains the weights of the trained model.')

    parser.add_argument('-d', '--dataset-file', type=str, default='datasets/2019-10-08-22-05-41.hdf5',
                        help='The name of the HDF5 file that contains the training and test datasets.')

    parser.add_argument('-r', '--rounds', type=int, default=10, choices=range(1, 1001), metavar="[1, 1000]",
                        help='The number of rounds to calculate the AUC.')

    parser.add_argument('-f', '--features', nargs='+', type=str, default=["camera"],
                        help="The name of the features in the HDF5 file.")

    parser.add_argument('-t', '--targets', nargs='+', type=str, default=["target"],
                        help="The name of the targets in the HDF5 file.")

    parser.add_argument('-u', '--usage-percentage', type=percent, default=100.0, metavar="[0, 100]",
                        help='The percentage of the dataset to use for a given experiment, which is a number in the '
                             'range [0, 100].')

    parser.add_argument('-sp', '--split-percentage', type=percent, default=50.0, metavar="[0, 100]",
                        help='The train/test split percentage, which is a number in the range [0, 100].')

    parser.add_argument('-i', '--interactive', action='store_true',
                        help="If this flag is passed, the user will be interactively asked to choose the file "
                             "containing the weights, else the last file (in terms of file name) in the folder "
                             "containing the trained model is used.")

    parser.add_argument('-s', '--save-auc', action='store_true',
                        help="If this flag is passed, the plot(s) containing the AUC scores are stored to a file.")

    parser.add_argument('-af', '--aucs-folder', type=str, default="models/{}/aucs".format(DEFAULT_MODEL),
                        help='The name of the folder where to save the AUC heat map.')

    return parser.parse_args()


def calculate_mean_auc(args, test_y, pred_y, masks):
    # A list of arrays of shape (len(args.targets), len(target_coordinates)), one for each round.
    auc_array = []

    print('Number of rounds = %d' % args.rounds)

    # Calculate the AUC for args.rounds time, then average the result.
    for _ in tqdm.tqdm(range(args.rounds)):

        # Calculate the AUC for each target coordinate and target combination, each time with the values of all
        # observations (or examples) in the test dataset.
        aucs = np.zeros((len(args.targets), len(target_coordinates)))

        for t, target in enumerate(args.targets):
            for c, target_coordinate in enumerate(target_coordinates):

                # Select the mask for the current target and coordinate, for all observations.
                mask = masks[target][:, c]

                if mask.dtype != np.dtype('bool'):
                    raise TypeError("mask is not a boolean array.")

                if np.sum(mask) == 0:
                    # If the mask only contains False elements, there is no label.
                    auc = 0.5
                else:
                    try:
                        # Compute the ROC-AUC score for coordinate c with all observations.
                        auc = roc_auc_score(test_y[target][mask, c].tolist(), pred_y[target][mask, c].tolist())
                    except ValueError:
                        auc = 0.5

                aucs[t][c] = auc

        auc_array.append(aucs)

    return np.mean(auc_array, axis=0)


def plot_auc_map(args, mean_auc):
    # One AUC map for each target.
    n = 17
    assert n * n == len(target_coordinates)
    auc_maps = np.zeros((len(args.targets), n, n))

    # Create the grid of target coordinates with the associated AUC value.
    for t, target in enumerate(args.targets):
        for c in range(len(target_coordinates)):
            x = 16 - (c // 17)
            y = c % 17
            auc_maps[t, x, y] = mean_auc[t, c]

    # Show the AUC maps.
    for target, auc_map in zip(args.targets, auc_maps):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
        auc_map = auc_map * 100
        sns.heatmap(auc_map.astype(int), cmap='gray', annot=True, vmin=50, vmax=100, fmt="d")
        sns.set(font_scale=0.9)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.title('Average ROC-AUC score for target="{}"'.format(target))

        if args.save_auc:
            if not path.exists(args.aucs_folder):
                os.makedirs(args.aucs_folder)
            file_path = path.join(args.aucs_folder, time.strftime("%Y-%m-%d-%H-%M-%S") + ".png")
            plt.savefig(file_path)

    plt.show()


def get_weights_file_name(args):
    if not path.exists(args.weights_folder):
        raise ValueError("{} does not exist.".format(args.weights_folder))

    files_names = [file_name for file_name in os.listdir(args.weights_folder) if file_name.endswith(".hdf5")]
    if len(files_names) == 0:
        raise ValueError("folder {} does not contain any .hdf5 file.".format(args.weights_folder))

    hdf5_index = len(files_names) - 1
    if args.interactive:
        print 'The following HDF5 files containing weights were found: '
        for j in range(len(files_names)):
            print files_names[j], "(index = {})".format(j)

        hdf5_index = int(raw_input('Please, enter the index of the desired HDF5 file: '))
        while hdf5_index < 0 or hdf5_index >= len(files_names):
            hdf5_index = int(raw_input('Please, enter the index of the desired HDF5 file: '))

    return files_names[hdf5_index]


def test(args=None, show_auc_map=True):
    if args is None:
        args = get_arguments()

    weights_file_name = get_weights_file_name(args)

    cnn = get_model()
    cnn.load_weights(path.join(args.weights_folder, weights_file_name))

    # masks is an array of the same shape as text_y, but with True in the position where there is a known label and
    # False in the positions where no label is known. masks is a (N, len(target_coordinates)) array, where N is the
    # number of observations or the size of the test dataset.
    test_x, test_y, masks = next(get_generator(hdf5_file_name=args.dataset_file, is_testset=True,
                                               usage_percentage=args.usage_percentage,
                                               split_percentage=args.split_percentage,
                                               features=args.features, targets=args.targets))

    loss = cnn.evaluate(test_x, test_y, verbose=1)
    print 'Test loss:', loss

    prediction = cnn.predict(test_x)

    # A dictionary from targets to their predictions.
    if isinstance(prediction, list):
        # If there are multiple targets, pred is a list of prediction, one for each target.
        pred_y = {target: p for target, p in zip(args.targets, prediction)}
    else:
        pred_y = {target: prediction for target in args.targets}

    mean_auc = calculate_mean_auc(args, test_y, pred_y, masks)

    if show_auc_map:
        plot_auc_map(args, mean_auc)

    return loss, prediction, mean_auc


def incrementally_test():
    args = get_arguments()

    models_folder = "models/2019-10-12-15-46-25"

    losses = []
    mean_aucs = []
    predictions = []

    for name in sorted(os.listdir(models_folder)):

        model_folder = path.join(models_folder, name)

        if path.isdir(models_folder):
            weights_folder = path.join(model_folder, "weights")
            aucs_folder = path.join(model_folder, "aucs")

            args.weights_folder = weights_folder
            args.aucs_folder = aucs_folder

            loss, prediction, mean_auc = test(args)

            losses.append(loss)
            predictions.append(prediction)
            mean_aucs.append(mean_auc)


if __name__ == '__main__':
    # test()
    incrementally_test()
