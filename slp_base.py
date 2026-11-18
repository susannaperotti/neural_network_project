"""Shared initialization, backpropagation, and mini-batch training."""

from abc import ABC, abstractmethod
from numbers import Integral, Real

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted, validate_data

from activations import logistic, logistic_derivative, relu, relu_derivative, tanh, tanh_derivative


class BaseSLPEstimator(BaseEstimator, ABC):
    """One-hidden-layer network trained with mini-batch SGD.

    max_iter counts epochs. batch_size=None uses the full training set per
    update. momentum=0 uses ordinary SGD. With early_stopping=True, reserve
    validation_fraction of the samples and stop after n_iter_no_change epochs
    without a validation loss improvement of more than tol. Restore the
    parameters with the lowest observed validation loss.
    """

    def __init__(
        self,
        hidden_layer_size=100,
        activation="logistic",
        learning_rate=0.001,
        max_iter=200,
        random_state=None,
        *,
        batch_size=8,
        early_stopping=False,
        validation_fraction=0.1,
        tol=1e-4,
        n_iter_no_change=10,
        momentum=0.0,
    ):
        self.hidden_layer_size = hidden_layer_size
        self.activation = activation
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.random_state = random_state
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.tol = tol
        self.n_iter_no_change = n_iter_no_change
        self.momentum = momentum

    def _get_activation_function(self):
        if self.activation == "relu":
            return relu, relu_derivative
        if self.activation == "tanh":
            return tanh, tanh_derivative
        if self.activation in ("logistic", "sigmoid"):
            return logistic, logistic_derivative
        if self.activation == "identity":
            return lambda z: z, np.ones_like
        raise ValueError(f"Unknown activation: {self.activation}")

    def _validate_parameters(self):
        self._get_activation_function()
        integer_params = {
            "hidden_layer_size": self.hidden_layer_size,
            "max_iter": self.max_iter,
            "n_iter_no_change": self.n_iter_no_change,
        }
        if self.batch_size is not None:
            integer_params["batch_size"] = self.batch_size
        for name, value in integer_params.items():
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("learning_rate", "validation_fraction", "tol", "momentum"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
                raise ValueError(f"{name} must be a finite real number")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be between 0 and 1")
        if self.tol < 0:
            raise ValueError("tol must be nonnegative")
        if not 0 <= self.momentum < 1:
            raise ValueError("momentum must be in [0, 1)")
        if not isinstance(self.early_stopping, (bool, np.bool_)):
            raise ValueError("early_stopping must be boolean")
        if self.random_state is not None and (
            not isinstance(self.random_state, Integral) or self.random_state < 0
        ):
            raise ValueError("random_state must be a nonnegative integer or None")

    def _initialize_weights(self, rng):
        # Glorot uniform initialization scales with each layer's fan-in and fan-out.
        fan_in = self.n_features_in_
        hidden = self.hidden_layer_size
        fan_out = self.n_outputs_
        hidden_limit = np.sqrt(6 / (fan_in + hidden))
        output_limit = np.sqrt(6 / (hidden + fan_out))
        self.W1_ = rng.uniform(-hidden_limit, hidden_limit, (fan_in, hidden))
        self.b1_ = np.zeros(hidden)
        self.W2_ = rng.uniform(-output_limit, output_limit, (hidden, fan_out))
        self.b2_ = np.zeros(fan_out)

    def _forward_propagation(self, X):
        activate, _ = self._get_activation_function()
        z1 = X @ self.W1_ + self.b1_
        a1 = activate(z1)
        z2 = a1 @ self.W2_ + self.b2_
        return z1, a1, z2, self._output_activation(z2)

    def _backward_propagation(self, X, y, z1, a1, z2, y_pred):
        _, activation_derivative = self._get_activation_function()
        output_error = self._output_error(y, y_pred)
        hidden_error = (output_error @ self.W2_.T) * activation_derivative(z1)
        return (
            X.T @ hidden_error,
            hidden_error.sum(axis=0),
            a1.T @ output_error,
            output_error.sum(axis=0),
        )

    @abstractmethod
    def _output_activation(self, logits):
        """Map output preactivations to predictions."""

    @abstractmethod
    def _output_error(self, y, y_pred):
        """Return the derivative of the batch loss with respect to logits."""

    @abstractmethod
    def _compute_loss(self, y_true, y_pred):
        """Compute the loss from targets and predictions."""

    def _loss_from_forward(self, y, logits, predictions):
        return self._compute_loss(y, predictions)

    def _loss_on(self, X, y):
        _, _, logits, predictions = self._forward_propagation(X)
        loss = self._loss_from_forward(y, logits, predictions)
        if not np.isfinite(loss):
            raise ValueError("Training diverged. Try scaling the data or reducing learning_rate.")
        return loss

    def _split_indices(self, y, rng):
        n_samples = len(y)
        if not self.early_stopping:
            return np.arange(n_samples), np.empty(0, dtype=int)
        n_validation = int(np.ceil(self.validation_fraction * n_samples))
        if n_validation >= n_samples:
            raise ValueError("Early stopping requires both training and validation samples")
        order = rng.permutation(n_samples)
        return order[n_validation:], order[:n_validation]

    def _train(self, X, y, rng, indices):
        train_indices, validation_indices = indices
        X_train, y_train = X[train_indices], y[train_indices]
        X_validation, y_validation = X[validation_indices], y[validation_indices]
        self.__dict__.pop("n_iter_", None)
        self._initialize_weights(rng)
        params = (self.W1_, self.b1_, self.W2_, self.b2_)
        velocities = [np.zeros_like(param) for param in params]
        self.loss_curve_ = []
        self.validation_loss_curve_ = []
        self.best_validation_loss_ = None
        patience_reference = np.inf
        stale_epochs = 0
        batch_size = self.batch_size or len(X_train)

        for _ in range(self.max_iter):
            order = rng.permutation(len(X_train))
            for start in range(0, len(X_train), batch_size):
                batch = order[start:start + batch_size]
                X_batch, y_batch = X_train[batch], y_train[batch]
                state = self._forward_propagation(X_batch)
                gradients = self._backward_propagation(X_batch, y_batch, *state)
                for param, velocity, gradient in zip(params, velocities, gradients):
                    velocity *= self.momentum
                    velocity -= self.learning_rate * gradient
                    param += velocity

            self.loss_curve_.append(self._loss_on(X_train, y_train))
            if self.early_stopping:
                loss = self._loss_on(X_validation, y_validation)
                self.validation_loss_curve_.append(loss)
                if self.best_validation_loss_ is None or loss < self.best_validation_loss_:
                    self.best_validation_loss_ = loss
                    best_params = [param.copy() for param in params]
                if loss < patience_reference - self.tol:
                    patience_reference, stale_epochs = loss, 0
                else:
                    stale_epochs += 1
                    if stale_epochs >= self.n_iter_no_change:
                        break

        self.n_iter_ = len(self.loss_curve_)
        if self.early_stopping:
            for param, best in zip(params, best_params):
                param[:] = best
        self.loss_ = self._loss_on(X_train, y_train)

    def _predict_outputs(self, X):
        check_is_fitted(self, "n_iter_")
        X = validate_data(self, X, reset=False, dtype=float)
        return self._forward_propagation(X)[-1]
