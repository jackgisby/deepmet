import os
import sys
import torch
import logging
import numpy as np

sys.path.append(os.path.join("..", "src"))
from utils.config import Config
from deepSVDD import DeepSVDD
from datasets.main import load_dataset


def visualise_deep_svdd(
    dataset_name="mol_key_test",
    net_name="cocrystal_transformer",
    xp_path="../log/mol_key_test",
    data_path="../data/mol_key_test",
    test_prefix=["zinc", "chembl"],
    load_config="../log/mol_key_test/config.json",
    load_model="../log/mol_key_test/model.tar",
    device="cuda",
    n_jobs_dataloader=0
):

    # Get configuration
    cfg = Config(locals().copy())
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()

    cfg.load_config(import_json=load_config)
    logger.info('Loaded configuration from %s.' % load_config)

    # Default device to 'cpu' if cuda is not available
    if not torch.cuda.is_available():
        print("cuda not available: defaulting to cpu")
        device = 'cpu'

    # Initialize DeepSVDD model and set neural network \phi
    deep_SVDD = DeepSVDD(cfg.settings['objective'], cfg.settings['nu'], cfg.settings['rep_dim'], cfg.settings['in_features'])
    deep_SVDD.set_network(cfg.settings['net_name'])

    deep_SVDD.load_model(model_path=load_model, load_ae=False)
    logger.info('Loading model from %s.' % load_model)

    for prefix in test_prefix:

        dataset, dataset_labels, val_dataset = load_dataset(dataset_name, data_path, prefix)

        deep_SVDD.test(dataset, device=device, n_jobs_dataloader=n_jobs_dataloader)

        deep_SVDD.visualise_network(dataset, device=device, n_jobs_dataloader=n_jobs_dataloader)

        indices, labels, latent = zip(*deep_SVDD.visualisation)
        indices, labels, latent = np.array(indices), np.array(labels), np.array(latent)

        latent[:, -1] = dataset_labels[indices, 2]
        np.savetxt(xp_path + '/latent.csv', latent, fmt="%s", delimiter=",")


if __name__ == '__main__':
    visualise_deep_svdd()
