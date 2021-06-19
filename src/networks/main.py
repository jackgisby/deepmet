from .basic_multilayer import BasicMultilayer, BasicMultilayerAutoencoder


def build_network(net_name):
    """Builds the neural network."""

    implemented_networks = ('basic_multilayer',)
    assert net_name in implemented_networks

    net = None

    if net_name == 'basic_multilayer':
        net = BasicMultilayer()

    return net


def build_autoencoder(net_name):
    """Builds the corresponding autoencoder network."""

    implemented_networks = ('basic_multilayer',)
    assert net_name in implemented_networks

    ae_net = None

    if net_name == 'basic_multilayer':
        ae_net = BasicMultilayerAutoencoder()

    return ae_net
