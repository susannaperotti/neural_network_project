"""Single-hidden-layer classifier with binary and multiclass outputs."""

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import validate_data

from activations import logistic, softmax
from slp_base import BaseSLPEstimator


class SimpleSLPClassifier(ClassifierMixin, BaseSLPEstimator):
    """Fit class probabilities with cross-entropy. Score returns accuracy."""

    def _output_activation(self, logits):
        if self.n_outputs_ == 1:
            return logistic(logits)
        return softmax(logits)

    def _output_error(self, y, y_pred):
        # With cross-entropy, the sigmoid/softmax derivative simplifies.
        return (y_pred - y) / len(y)

    def _compute_loss(self, y_true, y_pred):
        """Compute cross-entropy when only probabilities are available."""
        eps = np.finfo(float).eps
        probabilities = np.clip(y_pred, eps, 1 - eps)
        if probabilities.shape[1] == 1:
            terms = y_true * np.log(probabilities) + (1 - y_true) * np.log1p(-probabilities)
        else:
            terms = y_true * np.log(probabilities)
        return float(-terms.sum(axis=1).mean())

    def _loss_from_forward(self, y, logits, predictions):
        # Evaluate cross-entropy in logit space to retain precision on
        # confident predictions, even when probabilities have rounded.
        if logits.shape[1] == 1:
            signed_logits = np.where(y == 1, -logits, logits)
            return float(np.logaddexp(0, signed_logits).mean())
        shifted = logits - logits.max(axis=1, keepdims=True)
        target_logits = shifted[np.arange(len(y)), y.argmax(axis=1)]
        return float(np.mean(np.logaddexp.reduce(shifted, axis=1) - target_logits))

    def fit(self, X, y):
        """Fit to a feature matrix and one class label per sample."""
        self._validate_parameters()
        X, y = validate_data(self, X, y, dtype=float)
        if type_of_target(y) not in ("binary", "multiclass"):
            raise ValueError("Expected discrete binary or multiclass labels")

        classes, encoded = np.unique(y, return_inverse=True)
        n_outputs = 1 if len(classes) <= 2 else len(classes)
        if n_outputs == 1:
            targets = encoded[:, None].astype(float)
        else:
            targets = (encoded[:, None] == np.arange(n_outputs)).astype(float)

        rng = np.random.default_rng(self.random_state)
        indices = self._split_indices(targets, rng)
        self.classes_ = classes
        self.n_outputs_ = n_outputs
        self._train(X, targets, rng, indices)
        return self

    def predict_proba(self, X):
        """Return probabilities in the order of classes_."""
        probabilities = self._predict_outputs(X)
        if len(self.classes_) == 1:
            return np.ones_like(probabilities)
        if len(self.classes_) == 2:
            return np.column_stack((1 - probabilities[:, 0], probabilities[:, 0]))
        return probabilities

    def predict(self, X):
        """Return the most probable original class label for each sample."""
        probabilities = self.predict_proba(X)
        return self.classes_[probabilities.argmax(axis=1)]
