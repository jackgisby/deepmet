import os
import sys
import csv
import torch
import logging
import random
import numpy as np

sys.path.append(os.path.join("..", "src"))
from utils.config import Config
from deepSVDD import DeepSVDD
from datasets.main import load_dataset


def model_validation(cfg, val_dataset, net_name, pretrain, device, n_jobs_dataloader, ae_loss_function, n_tuning_rounds, tuneable_params):

    logger = logging.getLogger()

    all_params_tested = []
    all_params_scores = []
    for i in range(n_tuning_rounds):

        initialised_params = {}
        for param in tuneable_params.keys():
            initialised_params[param] = 10 ** (-1 * random.uniform(tuneable_params[param][0], tuneable_params[param][1]))

        all_params_tested.append(initialised_params)
        logger.info('For tuning round {}, testing parameters: {}'.format(str(i), str(initialised_params)))

        val_deep_SVDD = train_single_model(cfg, val_dataset, net_name, pretrain, device, n_jobs_dataloader, ae_loss_function, initialised_params)

        if len(all_params_scores) == 0 or all([val_deep_SVDD.results['test_loss'] < x for x in all_params_scores]):
            best_model = val_deep_SVDD

        all_params_scores.append(val_deep_SVDD.results['test_loss'])

    val, idx = min((val, idx) for (idx, val) in enumerate(all_params_scores))
    return all_params_tested, all_params_scores, idx, best_model


def train_single_model(cfg, dataset, net_name, pretrain, device, n_jobs_dataloader, ae_loss_function, model_params):

    logger = logging.getLogger()

    # Initialize DeepSVDD model and set neural network \phi
    deep_SVDD = DeepSVDD(cfg.settings['objective'], cfg.settings['nu'], cfg.settings["rep_dim"], cfg.settings['in_features'])
    deep_SVDD.set_network(net_name)

    logger.info('Pretraining: %s' % pretrain)
    if pretrain:

        # Log pretraining details
        logger.info('Pretraining optimizer: %s' % cfg.settings['optimizer_name'])
        logger.info('Pretraining learning rate: %g' % model_params['ae_lr'])
        logger.info('Pretraining epochs: %d' % cfg.settings['n_epochs'])
        logger.info('Pretraining batch size: %d' % cfg.settings['batch_size'])
        logger.info('Pretraining weight decay: %g' % cfg.settings['weight_decay'])

        # Pretrain model on dataset (via autoencoder)
        deep_SVDD.pretrain(dataset,
                           optimizer_name=cfg.settings['optimizer_name'],
                           lr=model_params['ae_lr'],
                           n_epochs=cfg.settings['n_epochs'],
                           lr_milestones=tuple(),
                           batch_size=cfg.settings['batch_size'],
                           weight_decay=cfg.settings['weight_decay'],
                           device=device,
                           n_jobs_dataloader=n_jobs_dataloader,
                           loss_function=ae_loss_function)

    # Log training details
    logger.info('Training optimizer: %s' % cfg.settings['optimizer_name'])
    logger.info('Training learning rate: %g' % model_params['lr'])
    logger.info('Training epochs: %d' % cfg.settings['n_epochs'])
    logger.info('Training batch size: %d' % cfg.settings['batch_size'])
    logger.info('Training weight decay: %g' % cfg.settings['weight_decay'])

    # Train model on dataset
    deep_SVDD.train(dataset,
                    optimizer_name=cfg.settings['optimizer_name'],
                    lr=model_params['lr'],
                    n_epochs=cfg.settings['n_epochs'],
                    lr_milestones=tuple(),
                    batch_size=cfg.settings['batch_size'],
                    weight_decay=cfg.settings['weight_decay'],
                    device=device,
                    n_jobs_dataloader=n_jobs_dataloader)

    # Test model
    deep_SVDD.test(dataset, device=device, n_jobs_dataloader=n_jobs_dataloader)

    return deep_SVDD


def run_deep_svdd(
    dataset_name="mol_key_test",
    net_name="cocrystal_transformer",
    xp_path="../log/mol_key_test",
    data_path="../data/mol_key_test",
    objective="soft-boundary",
    nu=0.1,
    device="cuda",  # "cuda"
    seed=1,
    optimizer_name="amsgrad",
    batch_size=2000,
    pretrain=False,
    n_jobs_dataloader=0,
    ae_loss_function="bce",
    in_features=2749,
    n_tuning_rounds=100,
    rep_dim=200,
    n_epochs=20,
    weight_decay=1e-5,
    tuneable_params={"lr": (3, 5)}
):
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
    dataset, dataset_labels, validation_dataset = load_dataset(dataset_name, data_path, "zinc", seed)
    logger.info('Parameters to be tuned are: %s' % tuneable_params)

    all_params, all_scores, final_idx, deep_SVDD = model_validation(cfg, validation_dataset, net_name, pretrain, device, n_jobs_dataloader, ae_loss_function, n_tuning_rounds, tuneable_params)
    final_params = all_params[final_idx]

    for param in final_params.keys():
        cfg.settings[param] = final_params[param]

    with open(xp_path + "/all_parameters.csv", "w", newline="") as all_parameters:
        all_parameters_csv = csv.writer(all_parameters)

        all_parameters_csv.writerow(list(all_params[0].keys()) + ["score"])

        assert len(all_params) == len(all_scores)
        for i in range(len(all_params)):
            all_parameters_csv.writerow(list(all_params[i].values()) + [all_scores[i]])

    logger.info('The parameters of the final model are: %s' % str(final_params))

    deep_SVDD.test(dataset, device=device, n_jobs_dataloader=n_jobs_dataloader)

    logger.info('The hypersphere radius is: %s' % str(deep_SVDD.R))

    # Save results, model, and configuration
    deep_SVDD.save_results(export_json=xp_path + '/results.json')
    deep_SVDD.save_model(export_model=xp_path + '/model.tar', save_ae=pretrain)
    cfg.save_config(export_json=xp_path + '/config.json')


if __name__ == '__main__':
    run_deep_svdd()
