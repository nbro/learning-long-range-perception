#!/usr/bin/env python
# -*- coding: utf-8 -*-


import argparse
import os
import time
from os import path

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from generator import get_generator
from model import get_model
from utils import percent


def get_arguments():
    parser = argparse.ArgumentParser(description="It trains a model, in a self-supervised fashion, given a HDF5 file "
                                                 "that contains the readings of a long-range and a short-range sensor, "
                                                 "as well as odometry info.")

    parser.add_argument('-m', '--model-folder', type=str, default=time.strftime("%Y-%m-%d-%H-%M-%S"),
                        help='The name of the folder that contains info about the trained model.')

    parser.add_argument('-f', '--models-folder', type=str, default="models",
                        help='The name of the folder that contains the trained models.')

    parser.add_argument('-d', '--dataset-file', type=str, default='datasets/2019-10-08-22-05-41.hdf5',
                        help='The name of the HDF5 file that contains the training and test datasets.')

    parser.add_argument('-u', '--usage-percentage', type=percent, default=100.0, metavar="[0, 100]",
                        help='The percentage of the dataset to use for a given experiment, which is a number in the '
                             'range [0, 100].')

    parser.add_argument('-sp', '--split-percentage', type=percent, default=50.0, metavar="[0, 100]",
                        help='The train/test split percentage, which is a number in the range [0, 100].')

    parser.add_argument('-fs', '--features', nargs='+', type=str, default=["camera"], metavar="t",
                        help="The name of the features in the HDF5 file.")

    parser.add_argument('-t', '--targets', nargs='+', type=str, default=["target"], metavar="t",
                        help="The name of the targets in the HDF5 file.")

    parser.add_argument('-e', '--epochs', type=int, default=2,
                        help='The number of epochs of the training phase.')

    parser.add_argument('-s', '--steps', type=int, default=1,
                        help='The number of training steps per epoch.')

    parser.add_argument('-bs', '--batch-size', type=int, default=64,
                        help='The size of the batches of the training data.')

    parser.add_argument('-lr', '--learning-rate', type=float, default=0.0002,
                        help='The learning rate used for the training phase')

    return parser.parse_args()


def plot_loss(history_df, show_plot=False, save_plot=True, loss_folder_path=None):
    if save_plot:
        if not isinstance(loss_folder_path, str):
            raise TypeError("loss_folder_path should be a string.")
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    for column in history_df:
        c = history_df[column]
        ax.plot(np.arange(len(c)), c)
        ax.legend(history_df.columns.values)

    if save_plot:
        plt.savefig(path.join(loss_folder_path, time.strftime("%Y-%m-%d-%H-%M-%S") + ".png"))

    if show_plot:
        plt.show()


def create_model_folders(args):
    if path.exists(args.model_folder):
        raise ValueError("{} already exists".format(args.model_folder))

    if not path.exists(args.models_folder):
        os.makedirs(args.models_folder)

    model_path = path.join(args.models_folder, args.model_folder)
    os.makedirs(model_path)

    weights_folder_path = path.join(model_path, "weights")
    if not path.exists(weights_folder_path):
        os.makedirs(weights_folder_path)

    weights_file_path = path.join(weights_folder_path, "{epoch:02d}-{val_loss:.4f}.hdf5")

    loss_folder_path = path.join(model_path, "loss")
    if not path.exists(loss_folder_path):
        os.makedirs(loss_folder_path)

    history_file_path = path.join(loss_folder_path, "history.csv")

    return weights_file_path, loss_folder_path, history_file_path


def train(args):
    if args is None:
        args = get_arguments()

    weights_file_path, loss_folder_path, history_file_path = create_model_folders(args)

    gen = get_generator(hdf5_file_name=args.dataset_file, features=args.features, targets=args.targets,
                        usage_percentage=args.usage_percentage, split_percentage=args.split_percentage,
                        batch_size=args.batch_size, is_testset=False, augment=True)

    val_x, val_y, _ = next(get_generator(hdf5_file_name=args.dataset_file, features=args.features,
                                         usage_percentage=args.usage_percentage, split_percentage=args.split_percentage,
                                         targets=args.targets, is_testset=True))

    cnn = get_model(learning_rate=args.learning_rate, show_summary=False)

    history = cnn.fit_generator(generator=gen, steps_per_epoch=args.steps, epochs=args.epochs,
                                validation_data=(val_x, val_y),
                                callbacks=[
                                    keras.callbacks.ModelCheckpoint(weights_file_path,
                                                                    save_best_only=True,
                                                                    save_weights_only=True),
                                    keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
                                ])

    history_df = pd.DataFrame(history.history, columns=history.history.keys())
    history_df.to_csv(history_file_path)
    plot_loss(history_df, save_plot=True, loss_folder_path=loss_folder_path)


def incrementally_train(usage_percentages=range(10, 101, 10)):
    models_folder = time.strftime("%Y-%m-%d-%H-%M-%S")
    args = get_arguments()
    for usage_percentage in usage_percentages:
        # Train a new model with the current usage_percentage.
        args.usage_percentage = usage_percentage
        args.model_folder = "{}/{}".format(models_folder, usage_percentage)
        train(args)


if __name__ == '__main__':
    # train()
    incrementally_train()
