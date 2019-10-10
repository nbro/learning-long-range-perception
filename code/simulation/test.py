#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
from os import path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tqdm
from sklearn.metrics import roc_auc_score

from generator import get_generator
from model import get_model
from settings import target_coordinates


def get_argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model-folder', type=str, default="models/2019-10-10-17-55-32/weights",
                        help='The name of the folder that contains the trained model.')

    parser.add_argument('-d', '--dataset-file', type=str, default='datasets/2019-10-08-22-05-41.hdf5',
                        help='The name of the HDF5 file that contains the training and test datasets.')

    parser.add_argument('-r', '--rounds', type=int, default=1, choices=range(1, 201),
                        help='The number of rounds')

    parser.add_argument('--features', nargs='+', type=str, default=["camera"], metavar="t",
                        help="The name of the features in the HDF5 file.")

    parser.add_argument('--targets', nargs='+', type=str, default=["target"], metavar="t",
                        help="The name of the targets in the HDF5 file.")

    return parser.parse_args()


def test():
    args = get_argument_parser()

    if not path.exists(args.model_folder):
        raise ValueError("{} does not exist.".format(args.model_folder))

    files_names = [file_name for file_name in os.listdir(args.model_folder) if file_name.endswith(".hdf5")]
    if len(files_names) == 0:
        raise ValueError("folder {} does not contain any .hdf5 file.".format(args.model_folder))

    cnn = get_model()

    print 'The following HDF5 files containing weights were found: '
    for j in range(len(files_names)):
        print files_names[j], "(index = {})".format(j)

    hdf5_index = int(raw_input('Please, enter the index of the desired HDF5 file: '))
    if hdf5_index < 0 or hdf5_index >= len(files_names):
        hdf5_index = len(files_names) - 1

    cnn.load_weights(path.join(args.model_folder, files_names[hdf5_index]))

    # mask is an array of the same shape as text_y, but with True in the position where there is a known label and False
    # in the positions where no label is known. masks is a (N, len(target_coordinates)) array, where N is the number of
    # observations or the size of the test dataset.
    test_x, test_y, masks = next(get_generator(hdf5_file_name=args.dataset_file, is_testset=True,
                                               features=args.features, targets=args.targets))

    loss = cnn.evaluate(test_x, test_y, verbose=1)
    print 'Test loss:', loss

    pred = cnn.predict(test_x)

    # A dictionary from targets to their predictions.
    if isinstance(pred, list):
        # If there are multiple targets, pred is a list of prediction, one for each target.
        prediction = {target: p for target, p in zip(args.targets, pred)}
    else:
        prediction = {target: pred for target in args.targets}

    # A list of arrays of shape (len(args.targets), len(target_coordinates)), one for each round.
    auc_array = []

    print('Number of rounds = %d' % args.rounds)

    for _ in tqdm.tqdm(range(args.rounds)):

        # Calculate the AUC for each target coordinate and target combination, each time with the values of all
        # observations (or examples) in the test dataset.
        aucs = np.zeros((len(args.targets), len(target_coordinates)))

        for i, target in enumerate(args.targets):
            for j, target_coordinate in enumerate(target_coordinates):

                mask = masks[target]

                if mask.dtype != np.dtype('bool'):
                    raise TypeError("mask is not a boolean array.")

                if np.sum(mask) == 0:
                    # If the mask only contains False elements, there is no label.
                    auc = 0.5
                else:
                    try:
                        assert test_y[target].shape == mask.shape
                        auc = roc_auc_score(test_y[target][mask].tolist(), prediction[target][mask].tolist())
                    except ValueError:
                        auc = 0.5

                aucs[i][j] = auc

        auc_array.append(aucs)

    mean_auc = np.mean(auc_array, axis=0)

    # One AUC map for each target.
    auc_maps = np.zeros((len(args.targets), 17, 17))

    for t, target in enumerate(args.targets):
        for k in range(len(target_coordinates)):
            i = 16 - (k // 17)
            j = k % 17
            auc_maps[t, i, j] = mean_auc[t, k]

    # Show the AUC maps.
    for target, auc_map in zip(args.targets, auc_maps):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 4))
        sns.heatmap(auc_map * 100, cmap='gray', annot=True, vmin=50, vmax=100)
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        plt.title('Average ROC-AUC score for target="{}"'.format(target))

    plt.show()


if __name__ == '__main__':
    test()
