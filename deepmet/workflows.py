#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2021 Ralf Weber
#
# This file is part of DeepMet.
#
# DeepMet is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DeepMet is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with DeepMet.  If not, see <https://www.gnu.org/licenses/>.
#
# This file incorporates work covered by the following copyright and
# permission notice:
#
#   Copyright (c) 2018 Lukas Ruff
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy
#   of this software and associated documentation files (the "Software"), to deal
#   in the Software without restriction, including without limitation the rights
#   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#   copies of the Software, and to permit persons to whom the Software is
#   furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in all
#   copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#   SOFTWARE.

import os
import torch
import random
import logging
import numpy as np

from deepmet.datasets import load_testing_dataset, load_training_dataset
from deepmet.auxiliary import get_fingerprints_from_meta, drop_selected_features, select_features
from deepmet.auxiliary import Config
from deepmet.core import DeepMet


def get_likeness_scores(dataset_path, results_path, load_config=None, load_model=None, device="cpu"):

    if load_model is None:
        print("Model not given, using pre-trained DeepMet model and config")

        load_model = os.path.join("", "data", "models", "deep_met_model.tar")
        load_config = os.path.join("", "data", "models", "config.json")

    cfg = Config(locals().copy())
    cfg.load_config(import_json=load_config)

    # Default device to 'cpu' if cuda is not available
    if not torch.cuda.is_available():
        print("cuda not available: defaulting to cpu")
        device = 'cpu'

    # If required, computes the fingerprints from the input smiles
    input_fingerprints_path = get_fingerprints_from_meta(dataset_path, os.path.join(results_path, "input_fingerprints.csv"))
    input_fingerprints_out_path = os.path.join(results_path, "input_fingerprints_processed.csv")

    # Filter features
    input_fingerprints_path = drop_selected_features(
        fingerprints_path=input_fingerprints_path,
        fingerprints_out_path=input_fingerprints_out_path,
        cols_to_remove=cfg.settings["selected_features"]
    )

    # Load data
    dataset, dataset_labels = load_testing_dataset(
        normal_dataset_path=input_fingerprints_path,
        normal_meta_path=dataset_path
    )

    deep_met_model = DeepMet(cfg.settings['objective'], cfg.settings['nu'], cfg.settings['rep_dim'], cfg.settings['in_features'])
    deep_met_model.set_network(cfg.settings['net_name'])

    deep_met_model.load_model(model_path=load_model)

    deep_met_model.test(dataset, device=device)

    test_scores = []
    for i in range(len(dataset_labels)):
        test_scores.append((dataset_labels[i][0], dataset_labels[i][1], deep_met_model.trainer.test_scores[i][2]))

    # ID, smiles, score
    return test_scores


def train_single_model(cfg, dataset):
    logger = logging.getLogger()

    if cfg.settings['seed'] != -1:
        random.seed(cfg.settings['seed'])
        np.random.seed(cfg.settings['seed'])
        torch.manual_seed(cfg.settings['seed'])
        logger.info('Set seed to %d.' % cfg.settings['seed'])

    # Initialize DeepMet model and set neural network \phi
    deep_met_model = DeepMet(cfg.settings["objective"], cfg.settings["nu"], cfg.settings["rep_dim"],
                             cfg.settings["in_features"])
    deep_met_model.set_network(cfg.settings["net_name"])

    # Log training details
    logger.info("Training optimizer: %s" % cfg.settings["optimizer_name"])
    logger.info("Training learning rate: %g" % cfg.settings["lr"])
    logger.info("Training epochs: %d" % cfg.settings["n_epochs"])
    logger.info("Training batch size: %d" % cfg.settings["batch_size"])
    logger.info("Training weight decay: %g" % cfg.settings["weight_decay"])

    # Train model on dataset
    deep_met_model.train(
        dataset,
        optimizer_name=cfg.settings["optimizer_name"],
        lr=cfg.settings["lr"],
        n_epochs=cfg.settings["n_epochs"],
        lr_milestones=cfg.settings["lr_milestones"],
        batch_size=cfg.settings["batch_size"],
        weight_decay=cfg.settings["weight_decay"],
        device=cfg.settings["device"]
    )

    # Test model
    deep_met_model.test(dataset, device=cfg.settings["device"])

    return deep_met_model


def train_likeness_scorer(
        normal_meta_path,
        results_path,
        non_normal_meta_path=None,
        normal_fingerprints_path=None,
        non_normal_fingerprints_path=None,
        net_name="cocrystal_transformer",
        objective="one-class",
        nu=0.1,
        rep_dim=200,
        device="cpu",
        seed=1,
        optimizer_name="amsgrad",
        lr=0.000100095,
        n_epochs=20,
        lr_milestones=tuple(),
        batch_size=2000,
        weight_decay=1e-5,
        validation_split=0.8,
        test_split=0.9
):
    """
    Train a DeepMet model based only on the 'normal' structures specified. 'non-normal' structures can be supplied
    to form a test set, however these are not used to train the model or optimise its parameters. The 'normal' and
    'non-normal' sets can be any classes of structures.
    """

    # Default device to 'cpu' if cuda is not available
    if not torch.cuda.is_available():
        print("cuda not available: defaulting to cpu")
        device = 'cpu'

    # If required, computes the fingerprints from the input smiles
    if normal_fingerprints_path is None:
        normal_fingerprints_path = get_fingerprints_from_meta(normal_meta_path,
                                                              os.path.join(results_path, "normal_fingerprints.csv"))

    normal_fingerprints_out_path = os.path.join(results_path, "normal_fingerprints_processed.csv")

    if non_normal_meta_path is None:
        non_normal_fingerprints_out_path = None

    else:
        if non_normal_fingerprints_path is None:
            non_normal_fingerprints_path = get_fingerprints_from_meta(non_normal_meta_path, os.path.join(results_path,
                                                                                                         "non_normal_fingerprints.csv"))

        non_normal_fingerprints_out_path = os.path.join(results_path, "non_normal_fingerprints_processed.csv")

    # Filter features if necessary
    normal_fingerprints_path, non_normal_fingerprints_paths, selected_features = select_features(
        normal_fingerprints_path=normal_fingerprints_path,
        normal_fingerprints_out_path=normal_fingerprints_out_path,
        non_normal_fingerprints_paths=non_normal_fingerprints_path,
        non_normal_fingerprints_out_paths=non_normal_fingerprints_out_path
    )

    non_normal_fingerprints_path = non_normal_fingerprints_paths[0]

    # Get configuration
    cfg = Config(locals().copy())

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    log_file = results_path + '/log.txt'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Print arguments
    logger.info('Log file is %s.' % log_file)
    logger.info('Export path is %s.' % results_path)
    logger.info('Network: %s' % net_name)

    logger.info('The filtered normal fingerprint matrix path is %s.' % normal_fingerprints_path)
    logger.info('The filtered normal meta is %s.' % normal_meta_path)

    if non_normal_meta_path is not None:
        logger.info('The filtered non-normal fingerprint matrix path is %s.' % non_normal_fingerprints_path)
        logger.info('The filtered non-normal meta is %s.' % non_normal_meta_path)
    else:
        logger.info('A non-normal set has not been supplied')

    # Print configuration
    logger.info('Deep SVDD objective: %s' % cfg.settings['objective'])
    logger.info('Nu-parameter: %.2f' % cfg.settings['nu'])

    logger.info('Computation device: %s' % device)

    # Load data
    dataset, dataset_labels, validation_dataset = load_training_dataset(
        normal_dataset_path=normal_fingerprints_path,
        normal_meta_path=normal_meta_path,
        non_normal_dataset_path=non_normal_fingerprints_path,
        non_normal_dataset_meta_path=non_normal_meta_path,
        seed=seed,
        validation_split=validation_split,
        test_split=test_split
    )

    cfg.settings["in_features"] = dataset.train_set.dataset.data.shape[1]
    logger.info('Number of input features: %d' % cfg.settings["in_features"])

    # Train the model (and estimate loss on the 'normal' validation set)
    deep_met_model = train_single_model(cfg, validation_dataset)

    # Test using separate test dataset (that ideally includes a set of 'non-normal' compounds)
    deep_met_model.test(dataset, device=device)

    logger.info('The AUC on the test dataset is: %s' % str(deep_met_model.results["test_auc"]))

    # close connection to log file
    logger.removeHandler(file_handler)

    # save model and config
    deep_met_model.save_model(os.path.join(results_path, "model.tar"))
    cfg.save_config(os.path.join(results_path, "config.json"))

    return deep_met_model