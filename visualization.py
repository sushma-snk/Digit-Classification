import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# PROBABILITY PLOT
# ============================================================

def plot_probabilities(
    probabilities,
    prediction
):

    digits = np.arange(
        10
    )


    fig, ax = plt.subplots(
        figsize=(10, 5)
    )


    bars = ax.bar(
        digits,
        probabilities
    )


    ax.set_xlabel(
        "Digit"
    )

    ax.set_ylabel(
        "Probability"
    )

    ax.set_title(
        "SVM Class Probabilities"
    )

    ax.set_xticks(
        digits
    )

    ax.set_ylim(
        0,
        1.05
    )


    for i, value in enumerate(
        probabilities
    ):

        ax.text(
            i,
            value + 0.02,
            f"{value:.2f}",
            ha="center"
        )


    bars[prediction].set_alpha(
        0.8
    )


    fig.tight_layout()

    return fig


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def plot_feature_importance(
    model,
    prediction
):

    # --------------------------------------------------------
    # SVC with RBF kernel does not have coef_
    # --------------------------------------------------------

    if not hasattr(
        model,
        "coef_"
    ):

        return None


    coefficients = \
        model.coef_[prediction]


    importance = np.abs(
        coefficients
    )


    importance = importance.reshape(
        28,
        28
    )


    fig, ax = plt.subplots(
        figsize=(6, 6)
    )


    ax.imshow(
        importance,
        cmap="hot"
    )


    ax.set_title(
        f"Pixel Importance for Digit {prediction}"
    )


    ax.axis(
        "off"
    )


    fig.tight_layout()

    return fig


# ============================================================
# CONFUSION MATRIX
# ============================================================

def plot_confusion_matrix(
    cm
):

    fig, ax = plt.subplots(
        figsize=(8, 7)
    )


    ax.imshow(
        cm,
        cmap="Blues"
    )


    ax.set_xlabel(
        "Predicted label"
    )

    ax.set_ylabel(
        "True label"
    )

    ax.set_title(
        "MNIST Confusion Matrix"
    )


    ax.set_xticks(
        range(10)
    )

    ax.set_yticks(
        range(10)
    )


    for i in range(10):

        for j in range(10):

            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=8
            )


    fig.tight_layout()

    return fig
