from .basic_multilayer import BasicMultilayer
from .cocrystal_transformer import CocrystalTransformer


def build_network(net_name, rep_dim, in_features):
    """Builds the neural network."""

    implemented_networks = ("basic_multilayer", "cocrystal_transformer")
    assert net_name in implemented_networks

    net = None

    if net_name == "basic_multilayer":
        net = BasicMultilayer(rep_dim=rep_dim, in_features=in_features)

    elif net_name == "cocrystal_transformer":
        net = CocrystalTransformer(rep_dim=rep_dim, in_features=in_features)

    return net


def build_autoencoder(net_name, rep_dim, in_features):
    """Builds the corresponding autoencoder network."""

    implemented_networks = ("basic_multilayer", "cocrystal_transformer")
    assert net_name in implemented_networks

    ae_net = None

    if net_name == "basic_multilayer":
        ae_net = BasicMultilayerAutoencoder(rep_dim=rep_dim, in_features=in_features)

    elif net_name == "cocrystal_transformer":
        ae_net = CocrystalTransformerAutoencoder(rep_dim=rep_dim, in_features=in_features)

    return ae_net
