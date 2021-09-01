from deepmet.networks.basic_multilayer import BasicMultilayer
from deepmet.networks.cocrystal_transformer import CocrystalTransformer


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
