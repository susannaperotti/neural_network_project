"""Reproduce the held-out comparison and save training_loss.png."""

from pathlib import Path

import matplotlib
import numpy as np
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler

from slp_classifier import SimpleSLPClassifier
from slp_regressor import SimpleSLPRegressor

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SEED = 42
HIDDEN_UNITS = 24
EPOCHS = 160
BATCH_SIZE = 16
MOMENTUM = 0.8
LEARNING_RATE = 0.02


def nonlinear_data():
    """Generate a smooth nonlinear target with independent Gaussian noise."""
    rng = np.random.default_rng(SEED)
    X = rng.uniform(-2, 2, size=(650, 3))
    x1, x2, x3 = X.T
    y = np.sin(2 * x1) + 0.5 * x2 * x3 + rng.normal(0, 0.1, size=len(X))
    return X, y


def compare(X, y, classification, *, demonstrate_early_stopping=False):
    stratify = y if classification else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=SEED, stratify=stratify
    )
    feature_scaler = StandardScaler()
    X_train = feature_scaler.fit_transform(X_train)
    X_test = feature_scaler.transform(X_test)

    common = dict(
        activation="tanh",
        max_iter=EPOCHS,
        batch_size=BATCH_SIZE,
        momentum=MOMENTUM,
        random_state=SEED,
    )
    own_class = SimpleSLPClassifier if classification else SimpleSLPRegressor
    reference_class = MLPClassifier if classification else MLPRegressor
    own = own_class(
        hidden_layer_size=HIDDEN_UNITS, learning_rate=LEARNING_RATE, **common
    ).fit(X_train, y_train)
    reference = reference_class(
        hidden_layer_sizes=(HIDDEN_UNITS,),
        solver="sgd",
        learning_rate_init=LEARNING_RATE,
        alpha=0.0,
        nesterovs_momentum=False,
        early_stopping=False,
        tol=0.0,
        n_iter_no_change=EPOCHS,
        **common,
    )

    if classification:
        reference.fit(X_train, y_train)
        reference_score = reference.score(X_test, y_test)
        reference_loss = np.asarray(reference.loss_curve_)
    else:
        target_scaler = StandardScaler()
        scaled_train = target_scaler.fit_transform(y_train[:, None]).ravel()
        scaled_test = target_scaler.transform(y_test[:, None]).ravel()
        reference.fit(X_train, scaled_train)
        reference_score = reference.score(X_test, scaled_test)
        # The reference objective uses half the squared error.
        reference_loss = 2 * np.asarray(reference.loss_curve_)

    early_result = None
    if demonstrate_early_stopping:
        early = own_class(
            hidden_layer_size=HIDDEN_UNITS,
            learning_rate=LEARNING_RATE,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=12,
            **common,
        ).fit(X_train, y_train)
        early_result = (
            early.score(X_test, y_test),
            early.n_iter_,
            int(np.argmin(early.validation_loss_curve_)) + 1,
        )
    return own.score(X_test, y_test), reference_score, own.loss_curve_, reference_loss, early_result


def main():
    experiments = [
        ("Diabetes", *load_diabetes(return_X_y=True), False),
        ("Breast cancer", *load_breast_cancer(return_X_y=True), True),
        ("Iris", *load_iris(return_X_y=True), True),
        ("Synthetic nonlinear regression", *nonlinear_data(), False),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
    print("| Dataset | Test metric | SimpleSLP | scikit-learn MLP |")
    print("| --- | --- | ---: | ---: |")

    for ax, (name, X, y, classification) in zip(axes.flat, experiments):
        own_score, reference_score, own_loss, reference_loss, early_result = compare(
            X, y, classification, demonstrate_early_stopping=name == "Diabetes"
        )
        if early_result is not None:
            diabetes_early_result = early_result
        metric = "Accuracy" if classification else "R²"
        print(f"| {name} | {metric} | {own_score:.4f} | {reference_score:.4f} |")
        ax.plot(range(1, len(own_loss) + 1), own_loss, label="SimpleSLP", color="#126781")
        ax.plot(
            range(1, len(reference_loss) + 1), reference_loss,
            label="scikit-learn MLP", color="#b44b21", linestyle="--",
        )
        ax.set_title(name, loc="left")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cross-entropy" if classification else "MSE (scaled targets)")
        ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False)

    score, stopped, best = diabetes_early_result
    print(f"\nDiabetes early stopping: R²={score:.4f}, stopped at epoch {stopped}, restored epoch {best}.")
    fig.suptitle("Training loss by dataset", fontsize=16)
    fig.savefig(Path(__file__).with_name("training_loss.png"), dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
