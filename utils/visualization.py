import matplotlib.pyplot as plt
import numpy as np


def plot_prediction_probabilities(
    probabilities,
    prediction
):

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    digits = np.arange(10)

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
        "CNN Output Probabilities"
    )

    ax.set_xticks(
        digits
    )

    ax.set_ylim(
        0,
        1.05
    )

    # Highlight predicted digit
    bars[prediction].set_alpha(
        0.8
    )

    for digit, probability in zip(
        digits,
        probabilities
    ):

        ax.text(
            digit,
            probability + 0.02,
            f"{probability:.2f}",
            ha="center",
            fontsize=9
        )

    fig.tight_layout()

    return fig


def plot_feature_maps(
    feature_maps,
    max_maps=16
):

    """
    Display feature maps produced by a CNN convolution layer.
    """

    number_of_maps = min(
        feature_maps.shape[-1],
        max_maps
    )

    columns = 4

    rows = int(
        np.ceil(
            number_of_maps /
            columns
        )
    )

    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(10, 2.5 * rows)
    )

    axes = np.array(
        axes
    ).reshape(-1)

    for i in range(
        number_of_maps
    ):

        axes[i].imshow(
            feature_maps[:, :, i],
            cmap="gray"
        )

        axes[i].set_title(
            f"Filter {i + 1}"
        )

        axes[i].axis(
            "off"
        )

    for i in range(
        number_of_maps,
        len(axes)
    ):

        axes[i].axis(
            "off"
        )

    fig.suptitle(
        "CNN Feature Maps",
        fontsize=16
    )

    fig.tight_layout()

    return fig
