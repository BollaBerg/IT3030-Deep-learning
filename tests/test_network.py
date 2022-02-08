import numpy as np

from src.activation_functions import Linear
from src.layer import Layer
from src.loss_functions import MeanSquaredError
from src.network import Network

def _create_network() -> Network:
    layer1 = Layer(
        2, 2, Linear(), (0, 1), 0.1
    )
    layer1.weights = np.array([
        [.15, .25], [.20, .30]
    ])
    layer1.biases = np.array([.35, .45])
    layer2 = Layer(2, 2, Linear(), (0, 1), 0.1)
    layer2.weights = np.array([
        [.40, .50], [.45, .55]
    ])
    layer2.biases = np.array([0.50, 0.60])
    network = Network(
        input_size=2,
        loss_function=MeanSquaredError(),
        weight_regularization=None,
        weight_regularization_rate=None,
        layers=[layer1, layer2],
        softmax=False
    )
    return network

def test_network_forward_pass():
    """Test that Network.forward_pass without Softmax gives correct result"""
    network = _create_network()
    network.softmax = None

    inputs = np.array([.05, .10])

    np.testing.assert_array_equal(
        network.forward_pass(inputs),
        network.layers[1].forward_pass(network.layers[0].forward_pass(inputs))
    )

    layer1 = np.array([
        inputs[0]*.15 + inputs[1]*.20 + .35,
        inputs[0]*.25 + inputs[1]*.30 + .45]
    )
    expected_output = np.array([
        layer1[0]*.40 + layer1[1]*.45 + .50,
        layer1[0]*.50 + layer1[1]*.55 + .60,
    ])

    np.testing.assert_array_equal(
        network.forward_pass(inputs),
        expected_output
    )

# def test_network_backward_pass():
#     """Test that Network.backward_pass works as expected"""
#     network = _create_network()
#     network.softmax = None
#     inputs = np.array([.05, .10])
#     predictions = network.forward_pass(inputs)
#     network.backward_pass(predictions, )

#     # Calculated by hand
#     expected_weights = np.array([
#         []
#     ])
#     expected_biases = np.array([

#     ])

#     np.testing.assert_array_equal(
#         network.layer[0].weights,
#         expected_weights
#     )
#     np.testing.assert_array_equal(
#         network.layers[0].biases,
#         expected_biases
#     )