"""
Activation Functions and Their Derivatives
"""

import numpy as np
from numpy.typing import NDArray


def relu(z: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    ReLU activation function.

    Parameters:
    -----------
    z : array-like
        Input values

    Returns:
    --------
    array-like
        Activated values
    """
    return np.maximum(0, z)


def relu_derivative(z: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Derivative of ReLU activation.

    Parameters:
    -----------
    z : array-like
        Input values (pre-activation)

    Returns:
    --------
    array-like
        Gradient values
    """
    return (np.asarray(z) > 0).astype(float)


def tanh(z: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Hyperbolic tangent activation function.

    Parameters:
    -----------
    z : array-like
        Input values

    Returns:
    --------
    array-like
        Activated values
    """
    return np.tanh(z)


def tanh_derivative(z: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Derivative of tanh activation.

    Parameters:
    -----------
    z : array-like
        Input values (pre-activation)

    Returns:
    --------
    array-like
        Gradient values
    """
    return 1 - np.tanh(z) ** 2


def logistic(z: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Logistic (sigmoid) activation function.

    Parameters:
    -----------
    z : array-like
        Input values

    Returns:
    --------
    array-like
        Activated values
    """
    return 1 / (1 + np.exp(-np.clip(z, -35.0, 35.0)))


def logistic_derivative(z: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Derivative of logistic activation.

    Parameters:
    -----------
    z : array-like
        Input values (pre-activation)

    Returns:
    --------
    array-like
        Gradient values
    """
    probability = logistic(z)
    return probability * (1 - probability)


def softmax(z: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Softmax activation for output layer (classification).

    Parameters:
    -----------
    z : array-like, shape (n_samples, n_classes)
        Input values

    Returns:
    --------
    array-like, shape (n_samples, n_classes)
        Probabilities that sum to 1 for each sample
    """
    shifted = z - np.max(z, axis=-1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum(axis=-1, keepdims=True)
