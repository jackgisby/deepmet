import click
from workflow.training import train_likeness_scorer
from workflow.scoring import get_likeness_scores


@click.command()
@click.argument("dataset_path", type=click.Path(exists=True), help="If using a pre-existing model, the examples to be classified, else this will be used as the 'normal' structure set.")
@click.argument("results_path", type=click.Path(exists=True), help="The path at which to save results.")
@click.option("--train_model", type=bool, default=False, help="Whether to train a new model or process smiles using an existing model.")
@click.argument("non_normal_dataset_path", type=click.Path(exists=True), default=None, help="If training a new model, this will form the 'non-self' class that the final model will be tested against.")
@click.option("--load_config", type=click.Path(exists=True), default=None, help="Config JSON-file path (default: None).")
@click.option("--load_model", type=click.Path(exists=True), default=None, help="Model file path (default: None).")
@click.option("--net_name", type=click.Choice(["cocrystal_transformer", "basic_multilayer"]), help="The model architecture to be used.")
@click.option("--objective", type=click.Choice(["one-class", "soft-boundary"]), default="one-class", help="Specify Deep SVDD objective ('one-class' or 'soft-boundary').")
@click.option("--nu", type=float, default=0.1, help="Deep SVDD hyperparameter nu (must be 0 < nu <= 1).")
@click.option("--device", type=str, default="cuda", help="Computation device to use ('cpu', 'cuda', 'cuda:2', etc.).")
@click.option("--seed", type=int, default=-1, help="Set seed. If -1, use randomization.")
@click.option("--optimizer_name", type=click.Choice(["adam", "amsgrad"]), default="adam", help="Name of the optimizer to use for Deep SVDD network training.")
@click.option("--lr", type=float, default=0.0001, help="Initial learning rate for Deep SVDD network training. Default=0.001")
@click.option("--n_epochs", type=int, default=20, help="Number of epochs to train.")
@click.option("--lr_milestone", type=int, default=0, multiple=True, help="Lr scheduler milestones at which lr is multiplied by 0.1. Can be multiple and must be increasing.")
@click.option("--batch_size", type=int, default=2000, help="Batch size for mini-batch training.")
@click.option("--weight_decay", type=float, default=1e-5, help="Weight decay (L2 penalty) hyperparameter for Deep SVDD objective.")
@click.option("--pretrain", type=bool, default=False, help="Pretrain neural network parameters via autoencoder.")
@click.option("--ae_optimizer_name", type=click.Choice(["adam", "amsgrad"]), default="adam", help="Name of the optimizer to use for autoencoder pretraining.")
@click.option("--ae_lr", type=float, default=0.0001, help="Initial learning rate for autoencoder pretraining. Default=0.001")
@click.option("--ae_n_epochs", type=int, default=20, help="Number of epochs to train autoencoder.")
@click.option("--ae_lr_milestone", type=int, default=0, multiple=True, help="Lr scheduler milestones at which lr is multiplied by 0.1. Can be multiple and must be increasing.")
@click.option("--ae_batch_size", type=int, default=2000, help="Batch size for mini-batch autoencoder training.")
@click.option("--ae_weight_decay", type=float, default=1e-5, help="Weight decay (L2 penalty) hyperparameter for autoencoder objective.")
@click.option("--n_jobs_dataloader", type=int, default=0, help="Number of workers for data loading. 0 means that the data will be loaded in the main process.")
@click.option("--isomeric_smiles", type=bool, default=True, help="If True, the smiles of the input dataset(s) will be converted to isomeric smiles using RDKit, else isomeric information will be discarded.")
def main(dataset_path, results_path, train_model, non_normal_dataset_path, load_config, load_model, net_name,
         objective, nu, device, seed, optimizer_name, lr, n_epochs, lr_milestone, batch_size, weight_decay, pretrain,
         ae_optimizer_name, ae_lr, ae_n_epochs, ae_lr_milestone, ae_batch_size, ae_weight_decay, n_jobs_dataloader):
    """
    Use or train a DeepSVDD model for the likeness scoring of compounds. In the case that the user is training a new
    model and metabolites are used as the 'self' dataset, as was done for the default pre-trained model, then resulting
    models will also be metabolite-likeness scorers. However, the 'self' dataset used as input can be any class of
    compounds, allowing the training of any compound-likeness scorer.

    If using a pre-trained model to carry out likeness scoring, the majority of arguments are ignored. The user must
    specify:
     - `dataset_path`
     - `results_path`
     - `train_model` (False)
     - `load_model`

     Otherwise, if `train_model` is `False`, then the majority of arguments are relevant. For argument descriptions
     and further detail see the CLI option help or the documentation for the `train_likeness_scorer` and
     `get_likeness_scores` functions.
    """

    if train_model:
        train_likeness_scorer(dataset_path=dataset_path, results_path=results_path,
                              load_config=load_config, non_normal_dataset_path=non_normal_dataset_path,
                              net_name=net_name, objective=objective, nu=nu, device=device, seed=seed,
                              optimizer_name=optimizer_name, lr=lr, n_epochs=n_epochs, lr_milestone=lr_milestone,
                              batch_size=batch_size, weight_decay=weight_decay, pretrain=pretrain,
                              ae_optimizer_name=ae_optimizer_name, ae_lr=ae_lr, ae_n_epochs=ae_n_epochs,
                              ae_lr_milestone=ae_lr_milestone, ae_batch_size=ae_batch_size,
                              ae_weight_decay=ae_weight_decay, n_jobs_dataloader=n_jobs_dataloader)

    else:
        get_likeness_scores(dataset_path=dataset_path, results_path=results_path, load_model=load_model)


if __name__ == "__main__":
    main()
