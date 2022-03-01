import src.activation_functions as fun

import numpy as np
from pytest import approx

def test_sigmoid_simple_value():
    """Test that Sigmoid function returns correct value for float"""
    sigmoid = fun.Sigmoid()
    assert sigmoid.apply(-1) == approx(0.2689414)
    assert sigmoid.apply(0) == 0.5
    assert sigmoid.apply(1) == approx(0.7310586)

def test_sigmoid_ndarray():
    """Test that Sigmoid function returns correct values for ndarray"""
    sigmoid = fun.Sigmoid()
    inputs = np.array([-1, 0, 1])
    np.testing.assert_array_almost_equal(
        sigmoid.apply(inputs),
        np.array([0.2689414, 0.5, 0.7310586])
    )

def test_sigmoid_derivative_simple_value():
    """Test that Sigmoid derivative returns correct value for floats"""
    sigmoid = fun.Sigmoid()
    assert sigmoid.derivative(-1) == approx(0.1966119332414818525374)
    assert sigmoid.derivative(0) == 0.25
    assert sigmoid.derivative(10) == approx(4.539580773595167103244e-5)

def test_sigmoid_derivative_ndarray():
    """Test that Sigmoid derivative returns correct values for ndarray"""
    sigmoid = fun.Sigmoid()
    inputs = np.array([-1, 0, 10])
    np.testing.assert_array_almost_equal(
        sigmoid.derivative(inputs),
        np.array([0.1966119332414818525374, 0.25, 4.539580773595167103244e-5])
    )


def test_tanh_simple_value():
    """Test that Tanh function returns correct value for float"""
    tanh = fun.Tanh()
    assert tanh.apply(-1) == approx(-0.7615941559557648881195)
    assert tanh.apply(0) == 0
    assert tanh.apply(5) == approx(0.999909204262595131211)

def test_tanh_ndarray():
    """Test that Tanh function returns correct values for ndarray"""
    tanh = fun.Tanh()
    inputs = np.array([-1, 0, 5])
    np.testing.assert_array_almost_equal(
        tanh.apply(inputs),
        np.array([-0.7615941559557648881195, 0, 0.999909204262595131211])
    )

def test_tanh_derivative_simple_value():
    """Test that Tanh derivative returns correct value for float"""
    tanh = fun.Tanh()
    assert tanh.derivative(-1) == approx(0.4199743416140260693945)
    assert tanh.derivative(0) == 1
    assert tanh.derivative(5) == approx(1.815832309438066841298E-4)

def test_tanh_derivative_ndarray():
    """Test that Tanh derivative returns correct values for ndarray"""
    tanh = fun.Tanh()
    inputs = np.array([-1, 0, 5])
    np.testing.assert_array_almost_equal(
        tanh.derivative(inputs),
        np.array([0.4199743416140260693945, 1, 1.815832309438066841298E-4])
    )


def test_relu_simple_value():
    """Test that Relu function returns correct value for float"""
    relu = fun.Relu()
    assert relu.apply(-1) == 0
    assert relu.apply(0) == 0
    assert relu.apply(1.5) == 1.5

def test_relu_ndarray():
    """Test that Relu function returns correct value for ndarray"""
    relu = fun.Relu()
    inputs = np.array([-1, 0, 1.5])
    np.testing.assert_array_equal(
        relu.apply(inputs),
        np.array([0, 0, 1.5])
    )

def test_relu_derivative_simple_value():
    """Test that Relu derivative returns correct value for float"""
    relu = fun.Relu()
    assert relu.derivative(-1) == 0
    assert relu.derivative(0) == 0
    assert relu.derivative(1.5) == 1

def test_relu_derivative_ndarray():
    """Test that Relu derivative returns correct value for ndarray"""
    relu = fun.Relu()
    inputs = np.array([-1, 0, 1.5])
    np.testing.assert_array_equal(
        relu.derivative(inputs), np.array([0, 0, 1])
    )


def test_linear_simple_value():
    """Test that Linear function returns correct value for float"""
    linear = fun.Linear()
    assert linear.apply(-1) == -1
    assert linear.apply(0) == 0
    assert linear.apply(10) == 10

def test_linear_ndarray():
    """Test that Linear function returns correct value for ndarray"""
    linear = fun.Linear()
    inputs = np.array([-1, 0, 10])
    np.testing.assert_array_equal(linear.apply(inputs), inputs)

def test_linear_derivative_simple_value():
    """Test that Linear derivative returns correct value for float"""
    linear = fun.Linear()
    assert linear.derivative(-1) == 1
    assert linear.derivative(0) == 1
    assert linear.derivative(10) == 1

def test_linear_derivative_ndarray():
    """Test that Linear derivative returns correct value for ndarray"""
    linear = fun.Linear()
    inputs = np.array([-1, 0, 10])
    np.testing.assert_array_equal(linear.derivative(inputs), np.ones(3))
