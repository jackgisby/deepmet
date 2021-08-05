import os
import torch
import logging
import random
import numpy as np

from utils.config import Config
from deepSVDD import DeepSVDD
from datasets.main import load_dataset
from utils.feature_processing import get_fingerprints_from_meta, select_features


def train_single_model(cfg, dataset, ae_loss_function=torch.nn.BCELoss(), seed=1):

    logger = logging.getLogger()

    if seed != -1:
        random.seed(cfg.settings['seed'])
        np.random.seed(cfg.settings['seed'])
        torch.manual_seed(cfg.settings['seed'])
        logger.info('Set seed to %d.' % cfg.settings['seed'])

    # Initialize DeepSVDD model and set neural network \phi
    deep_SVDD = DeepSVDD(cfg.settings["objective"], cfg.settings["nu"], cfg.settings["rep_dim"], cfg.settings["in_features"])
    deep_SVDD.set_network(cfg.settings["net_name"])

    logger.info("Pretraining: %s" % cfg.settings["pretrain"])
    if cfg.settings["pretrain"]:

        # Log pretraining details
        logger.info("Pretraining optimizer: %s" % cfg.settings["optimizer_name"])
        logger.info("Pretraining learning rate: %g" % cfg.settings["ae_lr"])
        logger.info("Pretraining epochs: %d" % cfg.settings["n_epochs"])
        logger.info("Pretraining batch size: %d" % cfg.settings["batch_size"])
        logger.info("Pretraining weight decay: %g" % cfg.settings["weight_decay"])

        # Pretrain model on dataset (via autoencoder)
        deep_SVDD.pretrain(
            dataset,
            optimizer_name=cfg.settings["optimizer_name"],
            lr=cfg.settings["ae_lr"],
            n_epochs=cfg.settings["n_epochs"],
            lr_milestones=cfg.settings["lr_milestones"],
            batch_size=cfg.settings["batch_size"],
            weight_decay=cfg.settings["weight_decay"],
            device=cfg.settings["device"],
            loss_function=ae_loss_function
        )

    # Log training details
    logger.info("Training optimizer: %s" % cfg.settings["optimizer_name"])
    logger.info("Training learning rate: %g" % cfg.settings["lr"])
    logger.info("Training epochs: %d" % cfg.settings["n_epochs"])
    logger.info("Training batch size: %d" % cfg.settings["batch_size"])
    logger.info("Training weight decay: %g" % cfg.settings["weight_decay"])

    # Train model on dataset
    deep_SVDD.train(
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
    deep_SVDD.test(dataset, device=cfg.settings["device"])

    return deep_SVDD


def train_likeness_scorer(
        normal_meta_path,
        results_path,
        load_config,
        non_normal_meta_path,
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
        pretrain,
        ae_optimizer_name,
        ae_lr,
        ae_n_epochs,
        ae_lr_milestones,
        ae_batch_size,
        ae_weight_decay,
        validation_split,
        test_split
):
    """
    Train a DeepSVDD model based only on the 'normal' structures specified. 'non-normal' structures can be supplied
    to form a test set, however these are not used to train the model or optimise its parameters. The 'normal' and
    'non-normal' sets can be any classes of structures.
    """

    # If required, computes the fingerprints from the input smiles
    if normal_fingerprints_path is None:
        normal_fingerprints_path = get_fingerprints_from_meta(normal_meta_path, os.path.join(results_path, "normal_fingerprints.csv"))

    normal_fingerprints_out_path = os.path.join(results_path, "normal_fingerprints_processed.csv")

    if non_normal_meta_path is None:
        non_normal_fingerprints_out_path = None

    else:
        if non_normal_fingerprints_path is None:
            non_normal_fingerprints_path = get_fingerprints_from_meta(non_normal_meta_path, os.path.join(results_path, "non_normal_fingerprints.csv"))

        non_normal_fingerprints_out_path = os.path.join(results_path, "non_normal_fingerprints_processed.csv")

    # Filter features if necessary
    normal_fingerprints_path, non_normal_fingerprints_path = select_features(
        normal_fingerprints_path=normal_fingerprints_path,
        normal_fingerprints_out_path=normal_fingerprints_out_path,
        non_normal_fingerprints_paths=non_normal_fingerprints_path,
        non_normal_fingerprints_out_paths=non_normal_fingerprints_out_path
    )

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

    # Default device to 'cpu' if cuda is not available
    if not torch.cuda.is_available():
        print("cuda not available: defaulting to cpu")
        device = 'cpu'

    logger.info('Computation device: %s' % device)

    # Load data
    dataset, dataset_labels, validation_dataset = load_dataset(
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
    deep_SVDD = train_single_model(cfg, validation_dataset, seed=seed)

    # Test using separate test dataset (that ideally includes a set of 'non-normal' compounds)
    deep_SVDD.test(dataset, device=device)

    logger.info('The AUC on the test dataset is: %s' % str(deep_SVDD.results["test_auc"]))

    # Save results, model, and configuration
    deep_SVDD.save_results(export_json=results_path + '/results.json')
    deep_SVDD.save_model(export_model=results_path + '/model.tar', save_ae=pretrain)
    cfg.save_config(export_json=results_path + '/config.json')

    return deep_SVDD
