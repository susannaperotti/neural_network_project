"""Numerical gradient and training-state checks for the SLP estimators."""

import numpy as np
import pytest

from slp_classifier import SimpleSLPClassifier
from slp_regressor import SimpleSLPRegressor


@pytest.mark.parametrize("task", ["binary", "multiclass", "multioutput"])
def test_gradients_match_finite_differences(task):
    X = np.array([[0.1, 0.8], [0.2, -0.7], [-0.3, 0.4]])
    if task == "binary":
        model = SimpleSLPClassifier()
        y = np.array([[0.0], [1.0], [0.0]])
    elif task == "multiclass":
        model = SimpleSLPClassifier()
        y = np.eye(3)
    else:
        model = SimpleSLPRegressor()
        y = np.array([[0.4, -0.2], [1.1, 0.1], [-0.5, 0.3]])

    model.set_params(hidden_layer_size=3, activation="tanh")
    model.n_features_in_ = X.shape[1]
    model.n_outputs_ = y.shape[1]
    model._initialize_weights(np.random.default_rng(4))
    analytic = model._backward_propagation(X, y, *model._forward_propagation(X))
    step = 1e-5

    for parameter, computed in zip((model.W1_, model.b1_, model.W2_, model.b2_), analytic):
        numerical = np.empty_like(parameter)
        for index in np.ndindex(parameter.shape):
            original = parameter[index]
            parameter[index] = original + step
            upper = model._loss_on(X, y)
            parameter[index] = original - step
            lower = model._loss_on(X, y)
            parameter[index] = original
            numerical[index] = (upper - lower) / (2 * step)
        np.testing.assert_allclose(computed, numerical, atol=1e-8)


def test_early_stopping_restores_lowest_validation_loss():
    rng = np.random.default_rng(12)
    X = rng.normal(size=(30, 3))
    y = 3 + X[:, 0] - X[:, 1]
    settings = dict(
        hidden_layer_size=4, learning_rate=0.03, random_state=5, batch_size=4,
        early_stopping=True, validation_fraction=0.2, tol=10, n_iter_no_change=3,
    )
    model = SimpleSLPRegressor(max_iter=50, **settings).fit(X, y)

    assert model.n_iter_ == len(model.loss_curve_) == len(model.validation_loss_curve_) == 4
    best_epoch = int(np.argmin(model.validation_loss_curve_)) + 1
    best = SimpleSLPRegressor(max_iter=best_epoch, **settings).fit(X, y)
    np.testing.assert_allclose(model.predict(X), best.predict(X))

    training_rows = np.random.default_rng(5).permutation(len(y))[6:]
    assert model.target_mean_[0] == pytest.approx(y[training_rows].mean())
    assert model.loss_ == pytest.approx(model.loss_curve_[best_epoch - 1])


def test_multiclass_probabilities_follow_original_label_order():
    X = [[-2], [-1], [0], [0.1], [1], [2]]
    y = ["third", "third", "first", "first", "second", "second"]
    model = SimpleSLPClassifier(
        hidden_layer_size=8, activation="tanh", learning_rate=0.1,
        max_iter=180, random_state=7, batch_size=None, momentum=0.8,
    ).fit(X, y)
    probabilities = model.predict_proba(X)
    np.testing.assert_array_equal(model.classes_, ["first", "second", "third"])
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    np.testing.assert_array_equal(model.predict(X), y)
