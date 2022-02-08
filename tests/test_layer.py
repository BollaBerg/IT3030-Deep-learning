import numpy as np

from src.activation_functions import Linear
from src.layer import Layer, SoftmaxLayer

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



def test_softmax_layer():
    """Test that Softmax.forward_pass produces correct output"""
    softmax = SoftmaxLayer()
    
    # All inputs taken from lecture slide
    np.testing.assert_array_almost_equal(
        softmax.forward_pass(np.array([1, 1, 2, 1, 1])),
        np.array([0.15, 0.15, 0.40, 0.15, 0.15]),
        decimal=2
    )
    np.testing.assert_array_almost_equal(
        softmax.forward_pass(np.array([-1, 0, 1, 0, -1])),
        np.array([0.07, 0.18, 0.50, 0.18, 0.07]),
        decimal=2
    )
    np.testing.assert_array_almost_equal(
        softmax.forward_pass(np.array([1, 2, 3, 4, 5])),
        np.array([0.01, 0.03, 0.08, 0.23, 0.64]),
        decimal=2
    )

def test_softmax_backward_pass():
    """Test that Softmax.backward_pass produces correct output"""
    softmax = SoftmaxLayer()

    output = softmax.forward_pass(np.array([1, 2, 3]))
    jacobi = softmax.backward_pass()

    expected_jacobi = np.array([
        [output[0]-output[0]**2, -output[0]*output[1], -output[0]*output[2]],
        [-output[1]*output[0], output[1]-output[1]**2, -output[1]*output[2]],
        [-output[2]*output[0], -output[2]*output[1], output[2]-output[2]**2]
    ])

    np.testing.assert_array_equal(jacobi, expected_jacobi)