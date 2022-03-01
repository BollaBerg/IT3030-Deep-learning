from pytest import approx
import numpy as np

from src.loss_functions import CrossEntropy, MeanSquaredError

def test_cross_entropy():
    """Test that CrossEntropy.apply gives correct outputs"""
    cross_entropy = CrossEntropy()

    # Values taken from lecture
    observations = np.array([.15, .35, .25, .25])
    targets = np.array([0, 0, 1, 0])
    assert cross_entropy.apply(observations, targets) == approx(2.0)

    observations = np.array([.05, .05, .9, .0])
    targets = np.array([0, 0, 1, 0])
    assert cross_entropy.apply(observations, targets) == approx(0.15, abs=1e-2)

    observations = np.array([.15, .35, .25, .25])
    targets = np.array([.8, .05, .05, .1])
    assert cross_entropy.apply(observations, targets) == approx(2.57, abs=1e-2)

    observations = np.array([.8, .1, .09, .01])
    targets = np.array([.8, .05, .05, .1])
    assert cross_entropy.apply(observations, targets) == approx(1.27, abs=1e-2)




def test_mean_squared_error():
    """Test that MeanSquaredError.apply gives correct outputs"""
    MSE = MeanSquaredError()

    observations = np.array([1, 2, 3, 4])
    targets = np.array([1, 4, 2, 0])
    assert MSE.apply(observations, targets) == 5.25

    observations = np.array([1.2, 0.5, 3.2, 4.2, 3.2])
    targets = np.array([1.5, 0.2, 3.9, 6.2, 5.2])
    assert MSE.apply(observations, targets) == 1.734


def test_mean_squared_error_derivatives():
    """Test that MeanSquaredError.derivative gives correct Jacobian"""
    MSE = MeanSquaredError()

    observations = np.array([1.2, 0.5, 3.2, 4.2, 3.2])
    targets = np.array([1.5, 0.2, 3.9, 6.2, 5.2])

    np.testing.assert_array_almost_equal(
        MSE.derivative(observations, targets),
        np.array([-0.12, 0.12, -0.28, -0.8, -0.8])
    )