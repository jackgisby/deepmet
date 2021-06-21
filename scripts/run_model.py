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
        net_name="basic_multilayer",
        xp_path="../log/mol_key_test",
        data_path="../data/mol_key_test",
        load_config=None,
        load_model=None,
        objective="one-class",
        nu=0.1,
        device="cpu",  # "cuda"
        seed=1,
        optimizer_name="amsgrad",
        lr=0.0001,
        n_epochs=150,
        lr_milestone=(50,),
        batch_size=200,
        weight_decay=0.5e-6,
        pretrain=True,
        ae_optimizer_name=None,
        ae_lr=0.00005,
        ae_n_epochs=100,
        ae_lr_milestone=(10, 50),
        ae_batch_size=None,
        ae_weight_decay=0.5e-3,
        n_jobs_dataloader=0,
        normal_class=1,
        ae_loss_function=torch.nn.BCELoss()
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
    logger.info('Normal class: %d' % normal_class)
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
        device = 'cpu'
    logger.info('Computation device: %s' % device)
    logger.info('Number of dataloader workers: %d' % n_jobs_dataloader)

    # Load data
    dataset, dataset_labels = load_dataset(dataset_name, data_path, normal_class)

    # Initialize DeepSVDD model and set neural network \phi
    deep_SVDD = DeepSVDD(cfg.settings['objective'], cfg.settings['nu'])
    deep_SVDD.set_network(net_name)

    # If specified, load Deep SVDD model (radius R, center c, network weights, and possibly autoencoder weights)
    if load_model:
        deep_SVDD.load_model(model_path=load_model, load_ae=True)
        logger.info('Loading model from %s.' % load_model)

    logger.info('Pretraining: %s' % pretrain)
    if pretrain:

        # Log pretraining details
        logger.info('Pretraining optimizer: %s' % cfg.settings['ae_optimizer_name'])
        logger.info('Pretraining learning rate: %g' % cfg.settings['ae_lr'])
        logger.info('Pretraining epochs: %d' % cfg.settings['ae_n_epochs'])
        logger.info('Pretraining learning rate scheduler milestones: %s' % (cfg.settings['ae_lr_milestone'],))
        logger.info('Pretraining batch size: %d' % cfg.settings['ae_batch_size'])
        logger.info('Pretraining weight decay: %g' % cfg.settings['ae_weight_decay'])

        # Pretrain model on dataset (via autoencoder)
        deep_SVDD.pretrain(dataset,
                           optimizer_name=cfg.settings['ae_optimizer_name'],
                           lr=cfg.settings['ae_lr'],
                           n_epochs=cfg.settings['ae_n_epochs'],
                           lr_milestones=cfg.settings['ae_lr_milestone'],
                           batch_size=cfg.settings['ae_batch_size'],
                           weight_decay=cfg.settings['ae_weight_decay'],
                           device=device,
                           n_jobs_dataloader=n_jobs_dataloader,
                           loss_function=ae_loss_function)

    # Log training details
    logger.info('Training optimizer: %s' % cfg.settings['optimizer_name'])
    logger.info('Training learning rate: %g' % cfg.settings['lr'])
    logger.info('Training epochs: %d' % cfg.settings['n_epochs'])
    logger.info('Training learning rate scheduler milestones: %s' % (cfg.settings['lr_milestone'],))
    logger.info('Training batch size: %d' % cfg.settings['batch_size'])
    logger.info('Training weight decay: %g' % cfg.settings['weight_decay'])

    # Train model on dataset
    deep_SVDD.train(dataset,
                    optimizer_name=cfg.settings['optimizer_name'],
                    lr=cfg.settings['lr'],
                    n_epochs=cfg.settings['n_epochs'],
                    lr_milestones=cfg.settings['lr_milestone'],
                    batch_size=cfg.settings['batch_size'],
                    weight_decay=cfg.settings['weight_decay'],
                    device=device,
                    n_jobs_dataloader=n_jobs_dataloader)

    # Test model
    deep_SVDD.test(dataset, device=device, n_jobs_dataloader=n_jobs_dataloader)

    # Plot most anomalous and most normal (within-class) test samples
    indices, labels, scores = zip(*deep_SVDD.results['test_scores'])
    indices, labels, scores = np.array(indices), np.array(labels), np.array(scores)
    idx_sorted = indices[labels == 0][np.argsort(scores[labels == 0])]  # sorted from lowest to highest anomaly score

    # record the most "normal" and the most outlying metabolites
    if dataset_name == "mol_key_test":

        label_normals = dataset_labels[idx_sorted[:32]]
        label_normals = np.append(label_normals, np.zeros([len(label_normals), 1]), 1)
        label_normals[:, -1] = np.array(scores[idx_sorted[:32] - min(indices)])
        np.savetxt(xp_path + '/normals.csv', label_normals, fmt="%s", delimiter=",")

        label_outliers = dataset_labels[idx_sorted[-32:]]
        label_outliers = np.append(label_outliers, np.zeros([len(label_outliers), 1]), 1)
        label_outliers[:, -1] = np.array(scores[idx_sorted[-32:] - min(indices)])
        np.savetxt(xp_path + '/outliers.csv', label_outliers, fmt="%s", delimiter=",")

    # Save results, model, and configuration
    deep_SVDD.save_results(export_json=xp_path + '/results.json')
    deep_SVDD.save_model(export_model=xp_path + '/model.tar', save_ae=pretrain)
    cfg.save_config(export_json=xp_path + '/config.json')


if __name__ == '__main__':
    run_deep_svdd()
