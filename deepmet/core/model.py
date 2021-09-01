import json
import torch

from deepmet.base.base_dataset import BaseADDataset
from deepmet.networks.network_builder import build_network
from deepmet.core.trainer import DeepMetTrainer


class DeepMet(object):
    """ Class for the DeepSVDD method adapted for compound anomaly detection. """

    def __init__(self, objective: str = 'one-class', nu: float = 0.1, rep_dim: int = 100, in_features: int = 2048):
        """Inits DeepMet with one of the two objectives and hyperparameter nu."""

        assert objective in ('one-class', 'soft-boundary'), "Objective must be either 'one-class' or 'soft-boundary'."
        self.objective = objective
        assert (0 < nu) & (nu <= 1), "For hyperparameter nu, it must hold: 0 < nu <= 1."
        self.nu = nu
        self.R = 0.0  # Hypersphere radius R
        self.c = None  # Hypersphere center c

        self.rep_dim = rep_dim
        self.in_features = in_features

        self.net_name = None
        self.net = None  # Neural network \phi

        self.trainer = None
        self.optimizer_name = None

        self.results = {
            'train_time': None,
            'test_auc': None,
            'test_time': None,
            'test_scores': None,
            'test_loss': None
        }

        self.visualisation = None

    def set_network(self, net_name):
        """ Builds the neural network \\phi. """

        self.net_name = net_name
        self.net = build_network(net_name, self.rep_dim, self.in_features)

    def train(self, dataset: BaseADDataset, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 50,
              lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda',
              n_jobs_dataloader: int = 0):
        """ Trains the DeepMet model on the training data. """

        self.optimizer_name = optimizer_name
        self.trainer = DeepMetTrainer(self.objective, self.R, self.c, self.nu, optimizer_name, lr=lr,
                                      n_epochs=n_epochs, lr_milestones=lr_milestones, batch_size=batch_size,
                                      weight_decay=weight_decay, device=device, n_jobs_dataloader=n_jobs_dataloader)

        # Get the model
        self.net = self.trainer.train(dataset, self.net)
        self.R = float(self.trainer.R.cpu().data.numpy())  # Get float
        self.c = self.trainer.c.cpu().data.numpy().tolist()  # Get list

        # Save results
        self.results['train_time'] = self.trainer.train_time
        self.results['R'] = self.R
        self.results['c'] = self.c

    def test(self, dataset: BaseADDataset, device: str = 'cuda', n_jobs_dataloader: int = 0):
        """ Tests the DeepMet model on the test data. """

        if self.trainer is None:
            self.trainer = DeepMetTrainer(self.objective, self.R, self.c, self.nu,
                                          device=device, n_jobs_dataloader=n_jobs_dataloader)

        # Test the model
        self.trainer.test(dataset, self.net)

        # Get results
        self.results['test_auc'] = self.trainer.test_auc
        self.results['test_time'] = self.trainer.test_time
        self.results['test_scores'] = self.trainer.test_scores
        self.results['test_loss'] = self.trainer.test_loss

    def visualise_network(self, dataset: BaseADDataset, device: str = 'cuda', n_jobs_dataloader: int = 0):
        """Gets the values of the model's latent layer for visualisation."""

        if self.trainer is None:
            self.trainer = DeepMetTrainer(self.objective, self.R, self.c, self.nu,
                                          device=device, n_jobs_dataloader=n_jobs_dataloader)

        self.trainer.visualise(dataset, self.net)
        self.visualisation = self.trainer.latent_visualisation

    def save_model(self, export_model):
        """Save DeepMet model to export_model."""

        net_dict = self.net.state_dict()

        torch.save({'R': self.R,
                    'c': self.c,
                    'net_dict': net_dict},
                   export_model)

    def load_model(self, model_path):
        """ Load DeepMet model from model_path. """

        model_dict = torch.load(model_path)

        self.R = model_dict['R']
        self.c = model_dict['c']
        self.net.load_state_dict(model_dict['net_dict'])

    def save_results(self, export_json):
        """ Save results dict to a JSON-file. """

        with open(export_json, 'w') as fp:
            json.dump(self.results, fp)
