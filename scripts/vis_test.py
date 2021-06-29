import os
import sys
import torch
import logging
import random
import numpy as np

sys.path.append(os.path.join("..", "src"))
from utils.config import Config
from deepSVDD import DeepSVDD
from datasets.main import load_dataset


def run_deep_svdd(
        dataset_name="mol_key_test",
        net_name="cocrystal_transformer",
        xp_path="../log/mol_key_test",
        data_path="../data/mol_key_test",
        load_config=None,
        load_model="../log/mol_key_test/model.tar",
        objective="soft-boundary",
        nu=0.1,
        device="cuda",  # "cuda"
        seed=1,
        optimizer_name="amsgrad",
        lr=0.00001,
        n_epochs=20,
        lr_milestone=tuple(),
        batch_size=500,
        weight_decay=1e-5,
        pretrain=True,
        ae_optimizer_name=None,
        ae_lr=0.00001,
        ae_n_epochs=None,
        ae_lr_milestone=None,
        ae_batch_size=None,
        ae_weight_decay=1e-3,
        n_jobs_dataloader=0,
        ae_loss_function="bce",
        rep_dim=150,
        in_features=2749
):
    # Set ae parameters based on regular parameters as default
    if ae_optimizer_name is None:
        ae_optimizer_name = optimizer_name

    if ae_lr is None:
        ae_lr = lr

    if ae_n_epochs is None:
        ae_n_epochs = n_epochs

    if ae_lr_milestone is None:
        ae_lr_milestone = lr_milestone

    if ae_batch_size is None:
        ae_batch_size = batch_size

    if ae_weight_decay is None:
        ae_weight_decay = weight_decay

    # Get configuration
    cfg = Config(locals().copy())

    if ae_loss_function == "bce":
        ae_loss_function = torch.nn.BCELoss()

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = xp_path + '/log.txt'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Print arguments
    logger.info('Log file is %s.' % log_file)
    logger.info('Data path is %s.' % data_path)
    logger.info('Export path is %s.' % xp_path)

    logger.info('Dataset: %s' % dataset_name)
    logger.info('Network: %s' % net_name)

    # If specified, load experiment config from JSON-file
    if load_config:
        cfg.load_config(import_json=load_config)
        logger.info('Loaded configuration from %s.' % load_config)

    # Print configuration
    logger.info('Deep SVDD objective: %s' % cfg.settings['objective'])
    logger.info('Nu-paramerter: %.2f' % cfg.settings['nu'])

    # Set seed
    if cfg.settings['seed'] != -1:
        random.seed(cfg.settings['seed'])
        np.random.seed(cfg.settings['seed'])
        torch.manual_seed(cfg.settings['seed'])
        logger.info('Set seed to %d.' % cfg.settings['seed'])

    # Default device to 'cpu' if cuda is not available
    if not torch.cuda.is_available():
        print("cuda not available: defaulting to cpu")
        device = 'cpu'

    logger.info('Computation device: %s' % device)
    logger.info('Number of dataloader workers: %d' % n_jobs_dataloader)

    # Load data
    dataset, dataset_labels = load_dataset(dataset_name, data_path)

    # Initialize DeepSVDD model and set neural network \phi
    deep_SVDD = DeepSVDD(cfg.settings['objective'], cfg.settings['nu'], cfg.settings['rep_dim'], cfg.settings['in_features'])
    deep_SVDD.set_network(net_name)

    # If specified, load Deep SVDD model (radius R, center c, network weights, and possibly autoencoder weights)
    if load_model:
        deep_SVDD.load_model(model_path=load_model, load_ae=False)
        logger.info('Loading model from %s.' % load_model)

    deep_SVDD.test(dataset, device=device, n_jobs_dataloader=n_jobs_dataloader)

    deep_SVDD.visualise_network(dataset, device=device, n_jobs_dataloader=n_jobs_dataloader)

    indices, labels, latent, latent_min_c, latent_min_c_sq, scores_unsq, scores = zip(*deep_SVDD.visualisation)
    indices, labels, latent, latent_min_c, latent_min_c_sq, scores_unsq, scores = np.array(indices), np.array(labels), np.array(latent), np.array(latent_min_c), np.array(latent_min_c_sq), np.array(scores_unsq), np.array(scores)

    latent[:, -1] = dataset_labels[indices, 2]
    np.savetxt(xp_path + '/latent.csv', latent, fmt="%s", delimiter=",")

    latent_min_c[:, -1] = dataset_labels[indices, 2]
    np.savetxt(xp_path + '/latent_min_c.csv', latent_min_c, fmt="%s", delimiter=",")

    latent_min_c_sq[:, -1] = dataset_labels[indices, 2]
    np.savetxt(xp_path + '/latent_min_c_sq.csv', latent_min_c_sq, fmt="%s", delimiter=",")

    scores_unsq[:, -1] = dataset_labels[indices, 2]
    np.savetxt(xp_path + '/scores_unsq.csv', scores_unsq, fmt="%s", delimiter=",")

    scores[:, -1] = dataset_labels[indices, 2]
    np.savetxt(xp_path + '/scores.csv', scores, fmt="%s", delimiter=",")


if __name__ == '__main__':
    run_deep_svdd()
