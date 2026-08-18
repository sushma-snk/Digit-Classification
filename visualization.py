import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# PROBABILITY PLOT
# ============================================================

def plot_probabilities(
    probabilities,
    prediction
):

    digits = np.arange(10)

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
        "Voting probability"
    )

    ax.set_title(
        "KNN Class Voting Distribution"
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
# NEAREST NEIGHBORS
# ============================================================

def plot_nearest_neighbors(
    images,
    labels,
    distances
):

    number = len(
        images
    )

    fig, axes = plt.subplots(
        1,
        number,
        figsize=(15, 3)
    )

    if number == 1:

        axes = [axes]

    for i in range(
        number
    ):

        image = images[i].reshape(
            28,
            28
        )

        axes[i].imshow(
            image,
            cmap="gray"
        )

        axes[i].set_title(
            f"Digit: {labels[i]}\n"
            f"Distance: {distances[i]:.2f}"
        )

        axes[i].axis(
            "off"
        )

    fig.suptitle(
        "Nearest MNIST Training Examples",
        fontsize=16
    )

    fig.tight_layout()

    return fig
