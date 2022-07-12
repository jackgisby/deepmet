#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2021 Jack Gisby, Ralf Weber
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

import os
import click

from deepmet.workflows import train_likeness_scorer, get_likeness_scores


@click.group()
def cli():
    """
    Command line application for training and applying compound-likeness models. Select
    a command for more information, for instance:

    deepmet train --help
    """
    pass


@cli.command()
@click.argument("normal_meta_path", type=click.Path(exists=True))
@click.option("--normal_fingerprints_path", type=click.Path(exists=True), default=None, help="Matrix of fingerprints corresponding to the rows of the file at NORMAL_META_PATH")
@click.option("--device", type=str, default="cuda", help="Computation device to use ('cpu', 'cuda', 'cuda:2', etc.)")
@click.option("--output_path", type=click.Path(exists=True), default=os.getcwd(), help="The path at which to save results")
@click.option("--load_model", type=click.Path(exists=True), default=None, help="The model file path. If this argument is not given, scoring will be performed with a pre-trained metabolite-likeness model")
@click.option("--load_config", type=click.Path(exists=True), default=None, help="A JSON-file path for the model configuration. If this argument is not given, the configuration for a pre-trained metabolite-likeness model will be loaded")
def score(
        normal_meta_path,
        output_path,
        normal_fingerprints_path,
        load_config,
        load_model,
        device
):
    """
    Generates likeness scores for the input compounds.

    NORMAL_META_PATH: Path of a file at which the input compounds are stored. Should
    be a CSV file, the first column of which has compound IDs and the second contains
    the SMILES molecular representations.

    A trained compound-likeness scorer can be specified using the `load_model` and
    `load_config` options, else a pre-trained metabolite-likeness model will be
    used.
    """
    get_likeness_scores(
        dataset_path=normal_meta_path,
        results_path=output_path,
        load_model=load_model,
        load_config=load_config,
        device=device,
        input_fingerprints_path=normal_fingerprints_path
    )


@cli.command()
@click.argument("normal_meta_path", type=click.Path(exists=True))
@click.option("--normal_fingerprints_path", type=click.Path(exists=True), default=None, help="Optional, matrix of fingerprints corresponding to the rows of the file at NORMAL_META_PATH")
@click.option("--non_normal_path", type=click.Path(exists=True), default=None, help="Will form the 'non-self' class that the final model will be tested against")
@click.option("--non_normal_fingerprints_path", type=click.Path(exists=True), default=None, help="Matrix of fingerprints corresponding to the rows of the file at non_normal_path")
@click.option("--device", type=str, default="cuda", show_default=True, help="Computation device to use ('cpu', 'cuda', 'cuda:2', etc.)")
@click.option("--output_path", type=click.Path(exists=True), default=os.getcwd(), help="The path at which to save results.")
@click.option("--net_name", type=click.Choice(["cocrystal_transformer", "basic_multilayer"]), default="cocrystal_transformer", help="The model architecture to be used")
@click.option("--objective", type=click.Choice(["one-class", "soft-boundary"]), default="one-class", help="Deep SVDD objective")
@click.option("--nu", type=float, default=0.1, show_default=True, help="The hyperparameter nu (must be 0 < nu <= 1)")
@click.option("--rep_dim", type=int, default=200, show_default=True, help="The number of features used in the representation dimension of the model")
@click.option("--seed", type=int, default=-1, show_default=True, help="Set seed. If -1, use randomisation")
@click.option("--optimizer_name", type=click.Choice(["adam", "amsgrad"]), default="adam", help="Name of the optimizer to use for Deep SVDD network training")
@click.option("--lr", type=float, default=0.0001, show_default=True, help="Initial learning rate for Deep SVDD network training")
@click.option("--n_epochs", type=int, default=20, show_default=True, help="Number of epochs to train")
@click.option("--lr_milestones", type=int, default=tuple(), multiple=True, help="Lr scheduler milestones at which lr is multiplied by 0.1. Can be multiple and must be increasing")
@click.option("--batch_size", type=int, default=2000, show_default=True, help="Batch size for mini-batch training")
@click.option("--weight_decay", type=float, default=1e-5, show_default=True, help="Weight decay (L2 penalty) hyperparameter for Deep SVDD objective")
@click.option("--validation_split", type=float, default=0.8, show_default=True, help="The percentile at which to split the training and validation set")
@click.option("--test_split", type=float, default=0.9, show_default=True, help="The percentile at which to split the validation and the test set")
@click.option("--no_filter", type=bool, is_flag=True, default=False, help="If flag is provided, no feature selection will be performed on the inputs")
def train(
        normal_meta_path,
        output_path,
        non_normal_path,
        normal_fingerprints_path,
        non_normal_fingerprints_path,
        net_name,
        objective,
        nu,
        rep_dim,
        device,
        seed,
        optimizer_name,
        lr,
        n_epochs,
        lr_milestones,
        batch_size,
        weight_decay,
        validation_split,
        test_split,
        no_filter
):
    """
    Trains a likeness scorer for the input compounds.

    NORMAL_META_PATH: Path of a file at which the input compounds are stored. Should
    be a CSV file, the first column of which has compound IDs and the second contains
    the SMILES molecular representations.
    """

    train_likeness_scorer(
        normal_meta_path=normal_meta_path,
        results_path=output_path,
        non_normal_meta_path=non_normal_path,
        normal_fingerprints_path=normal_fingerprints_path,
        non_normal_fingerprints_path=non_normal_fingerprints_path,
        net_name=net_name,
        objective=objective,
        nu=nu,
        rep_dim=rep_dim,
        device=device,
        seed=seed,
        optimizer_name=optimizer_name,
        lr=lr,
        n_epochs=n_epochs,
        lr_milestones=lr_milestones,
        batch_size=batch_size,
        weight_decay=weight_decay,
        validation_split=validation_split,
        test_split=test_split,
        filter_features=not no_filter
    )


if __name__ == "__main__":
    cli()
