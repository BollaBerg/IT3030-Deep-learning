from dataclasses import dataclass
import numpy as np

from src.activation_functions import Linear
from src.layer import Layer
from src.loss_functions import MeanSquaredError
from src.network import Network

LEARNING_RATE = 0.1

class MockImageClass:
    def one_hot(self):
        return np.array([0, 1])

@dataclass
class MockImage:
    data : np.ndarray
    image_class : MockImageClass


def _create_network() -> Network:
    layer1 = Layer(
        2, 2, Linear(), (0, 1), LEARNING_RATE
    )
    layer1.weights = np.array([
        [.15, .25], [.20, .30]
    ])
    layer1.biases = np.array([.35, .45])
    layer2 = Layer(2, 2, Linear(), (0, 1), LEARNING_RATE)
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

def test_network_backward_pass():
    """Test that Network.backward_pass works as expected"""
    network = _create_network()
    network.softmax = None
    inputs = np.array([.05, .10])
    predictions = network.forward_pass(inputs)
    targets = np.array([0.01, 0.99])
    network.backward_pass(predictions, targets)

    layer1 = np.array([
        inputs[0]*.15 + inputs[1]*.20 + .35,
        inputs[0]*.25 + inputs[1]*.30 + .45]
    )
    expected_output = np.array([
        layer1[0]*.40 + layer1[1]*.45 + .50,
        layer1[0]*.50 + layer1[1]*.55 + .60,
    ])

    # Calculated by hand
    l2 = network.layers[1]
    expected_weights_l2 = np.array([
        [.40 - LEARNING_RATE * (
            network.loss_function.derivative(predictions, targets)[0]
            * l2.activation_function.derivative(expected_output[0])
            * layer1[0]
        ),
         .50 - LEARNING_RATE * (
            network.loss_function.derivative(predictions, targets)[1]
            * l2.activation_function.derivative(expected_output[1])
            * layer1[0]
        )],
        [.45 - LEARNING_RATE * (
            network.loss_function.derivative(predictions, targets)[0]
            * l2.activation_function.derivative(expected_output[0])
            * layer1[1]
        ),
         .55 - LEARNING_RATE * (
            network.loss_function.derivative(predictions, targets)[1]
            * l2.activation_function.derivative(expected_output[1])
            * layer1[1]
        ),]
    ])
    expected_biases_l2 = np.array([
        .50 - LEARNING_RATE * (
            network.loss_function.derivative(predictions, targets)[0]
        ),
        .60 - LEARNING_RATE * (
            network.loss_function.derivative(predictions, targets)[1]
        ),
    ])

    np.testing.assert_array_equal(
        network.layers[1].weights,
        expected_weights_l2
    )
    np.testing.assert_array_equal(
        network.layers[1].biases,
        expected_biases_l2
    )

def test_network_gets_closer():
    """Test that Network gets closer to the target by training"""
    network = _create_network()
    dataset = [
        MockImage(np.array([0.05, 0.10]), MockImageClass())
        for _ in range(500)
    ]
    original_prediction = network.predict(np.array([0.05, 0.10]))
    network.train(dataset)
    later_prediction = network.predict(np.array([0.05, 0.10]))
    target = np.array([0.01, 0.99])

    assert abs(later_prediction[0] - target[0]) < abs(original_prediction[0] - target[0])
    assert abs(later_prediction[1] - target[1]) < abs(original_prediction[1] - target[1])