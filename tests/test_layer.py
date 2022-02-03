import numpy as np

from src.activation_functions import Linear
from src.layer import Layer

def _create_layer():
    return Layer(
        size=3, input_size=2,
        activation_function=Linear(),
        initial_weight_range=(0.2, 0.8),
        learning_rate=0.1,
        bias_range=(0.1, 0.9)
    )

def test_layer_creation():
    """Test that Layer.__init__ produces correct instance"""
    layer = _create_layer()
    assert layer.weights.shape == (2, 3)

    assert isinstance(layer.activation_function, Linear)

    assert np.min(layer.weights) >= 0.2
    assert np.max(layer.weights) <= 0.8

    assert layer.learning_rate == 0.1

    assert np.min(layer.biases) >= 0.1
    assert np.max(layer.biases) <= 0.9

def test_layer_forward_pass():
    """Test that Layer.forward_pass gives correct output"""
    layer = _create_layer()

    layer.weights = np.array([
        [1, 2, 3],
        [4, 5, 6]
    ])
    layer.biases = np.array([0, 0, 0])
    inputs = np.array([1, 100])

    np.testing.assert_array_equal(
        layer.forward_pass(inputs),
        np.array([1*1 + 4*100, 2*1 + 5*100, 3*1 + 6*100])
    )

def test_layer_forward_pass_with_bias():
    """Test that Layer.forward_pass applies biases correctly"""
    layer = _create_layer()

    layer.weights = np.array([
        [1, 2, 3],
        [4, 5, 6]
    ])
    layer.biases = np.array([42, 43, 44])
    inputs = np.array([1, 100])

    np.testing.assert_array_equal(
        layer.forward_pass(inputs),
        np.array([1*1 + 4*100 + 42, 2*1 + 5*100 + 43, 3*1 + 6*100 + 44])
    )
