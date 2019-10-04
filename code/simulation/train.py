#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Train the neural network model using the given training set."""

import argparse
import os
import time
from os import path

import keras

from generator import generator
from model import model


def get_argument_parser():
    parser = argparse.ArgumentParser(description="It trains a CNN, in a self-supervised fashion, given a HDF5 file "
                                                 "that contains the readings of a long-range and a short-range sensor, "
                                                 "as well as odometry info.")

    parser.add_argument('-m', '--model-folder', type=str, default=time.strftime("%Y-%m-%d-%H-%M-%S"),
                        help='The name of the folder that contains info about the trained model.')

    parser.add_argument('-f', '--models-folder', type=str, default="models",
                        help='The name of the folder that contains the trained models.')

    parser.add_argument('-d', '--dataset-file', type=str, default='dataset.hdf5',
                        help='The name of the HDF5 file that contains the training and test datasets.')

    parser.add_argument('-e', '--epochs', type=int, default=100,
                        help='The number of epochs of the training phase.')

    parser.add_argument('-s', '--steps', type=int, default=1000,
                        help='The number of training steps per epoch.')

    # See section V.C of the paper "Learning Long-Range Perception Using Self-Supervision from Short-Range Sensors and
    # Odometry" (https://arxiv.org/pdf/1809.07207.pdf).
    parser.add_argument('-sp', '--split-percentage', type=float, default=50.0, choices=range(0, 101),
                        help='The train/test split percentage, which is a number in the range [0, 100].')

    parser.add_argument('-bs', '--batch-size', type=int, default=64,
                        help='The size of the batches of the training data.')

    parser.add_argument('-lr', '--learning-rate', type=float, default=0.0002,
                        help='The learning rate used for the training phase')

    return parser.parse_args()


def train():
    """Train the neural network model, save the weights and show the learning error over time."""
    args = get_argument_parser()
    print args

    if path.exists(args.model_folder):
        raise ValueError("{} already exists".format(args.model_folder))

    if not path.exists(args.models_folder):
        os.makedirs(args.models_folder)

    model_path = path.join(args.models_folder, args.model_folder)
    os.makedirs(model_path)

    weights_folder_path = path.join(model_path, "weights")
    if not path.exists(weights_folder_path):
        os.makedirs(weights_folder_path)
    weights_file_path = path.join(weights_folder_path, "{epoch:02d}-{val_loss:.4f}.cp")

    cnn = model(args.learning_rate, show_summary=False)

    gen = generator(split_percentage=args.split_percentage, hdf5_file_name=args.dataset_file,
                    batch_size=args.batch_size,
                    is_testset=False, augment=True)

    val_x, val_y, _ = next(generator(hdf5_file_name=args.dataset_file, is_testset=True, testset_index=-1))

    history = cnn.fit_generator(generator=gen, steps_per_epoch=args.steps, epochs=args.epochs,
                                validation_data=(val_x, val_y),
                                callbacks=[
                                    keras.callbacks.ModelCheckpoint(weights_file_path,
                                                                    save_best_only=True,
                                                                    save_weights_only=True),
                                    keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
                                ])


if __name__ == '__main__':
    train()
