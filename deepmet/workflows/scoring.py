import os
import torch

from deepmet.utils.config import Config
from deepmet.core.model import DeepMet
from deepmet.datasets.load_data import load_testing_dataset
from deepmet.utils.feature_processing import get_fingerprints_from_meta, drop_selected_features


def get_likeness_scores(dataset_path, results_path, load_config=None, load_model=None, device="cpu"):

    if load_model is None:
        print("Model not given, using pre-trained DeepMet model and config")

        load_model = os.path.join("..", "data", "models", "deep_met_model.tar")
        load_config = os.path.join("..", "data", "models", "config.json")

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
