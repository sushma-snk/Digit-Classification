import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from PIL import ImageOps
from PIL import ImageFilter


def preprocess_image(image):
    """
    Convert an uploaded/camera image into a 28x28 MNIST-style image.

    Steps:
    1. Convert to grayscale
    2. Improve contrast
    3. Detect foreground/background
    4. Crop the digit
    5. Resize while preserving aspect ratio
    6. Center the digit in a 28x28 image

    Returns:
        processed_image
        debug_images
    """

    original = image.copy()

    # ========================================================
    # 1. GRAYSCALE
    # ========================================================

    gray = ImageOps.grayscale(image)

    gray = ImageOps.autocontrast(gray)

    # ========================================================
    # 2. CONVERT TO ARRAY
    # ========================================================

    arr = np.asarray(
        gray
    ).astype(
        np.uint8
    )

    # ========================================================
    # 3. ESTIMATE BACKGROUND
    # ========================================================

    border_pixels = np.concatenate(
        [
            arr[0, :],
            arr[-1, :],
            arr[:, 0],
            arr[:, -1],
        ]
    )

    background_mean = float(
        np.mean(
            border_pixels
        )
    )

    center_mean = float(
        np.mean(
            arr
        )
    )

    # ========================================================
    # 4. INVERT IMAGE IF NECESSARY
    # ========================================================

    if background_mean > center_mean:

        # Dark digit on light background

        inverted = ImageOps.invert(
            gray
        )

    else:

        # Already light digit on dark background

        inverted = gray

    # ========================================================
    # 5. SLIGHT BLUR
    # ========================================================

    inverted = inverted.filter(
        ImageFilter.GaussianBlur(
            radius=0.4
        )
    )

    arr2 = np.asarray(
        inverted
    )

    # ========================================================
    # 6. FIND DIGIT REGION
    # ========================================================

    threshold = max(
        20,
        np.percentile(
            arr2,
            70
        )
    )

    mask = (
        arr2 > threshold
    )

    if np.any(mask):

        ys, xs = np.where(
            mask
        )

        x0 = xs.min()
        x1 = xs.max()

        y0 = ys.min()
        y1 = ys.max()

        # ----------------------------------------------
        # Add padding around digit
        # ----------------------------------------------

        digit_width = (
            x1 - x0 + 1
        )

        digit_height = (
            y1 - y0 + 1
        )

        padding = max(
            4,
            int(
                0.15 *
                max(
                    digit_width,
                    digit_height
                )
            )
        )

        x0 = max(
            0,
            x0 - padding
        )

        y0 = max(
            0,
            y0 - padding
        )

        x1 = min(
            arr2.shape[1] - 1,
            x1 + padding
        )

        y1 = min(
            arr2.shape[0] - 1,
            y1 + padding
        )

        cropped = inverted.crop(
            (
                x0,
                y0,
                x1 + 1,
                y1 + 1,
            )
        )

    else:

        cropped = inverted

    # ========================================================
    # 7. RESIZE TO 20 × 20
    # ========================================================

    cropped.thumbnail(
        (
            20,
            20
        ),
        Image.Resampling.LANCZOS
    )

    # ========================================================
    # 8. CREATE 28 × 28 CANVAS
    # ========================================================

    canvas = Image.new(
        "L",
        (
            28,
            28
        ),
        0
    )

    # ========================================================
    # 9. CENTER DIGIT
    # ========================================================

    left = (
        28 -
        cropped.width
    ) // 2

    top = (
        28 -
        cropped.height
    ) // 2

    canvas.paste(
        cropped,
        (
            left,
            top
        )
    )

    processed_image = canvas

    # ========================================================
    # DEBUG IMAGES
    # ========================================================

    debug_images = {

        "Original":
            original,

        "Grayscale":
            gray,

        "Inverted / foreground":
            inverted,

        "Cropped":
            cropped,

        "Centered 28×28":
            processed_image,

    }

    return (
        processed_image,
        debug_images
    )


# ============================================================
# PREPROCESSING VISUALIZATION
# ============================================================

def make_debug_figure(
    original,
    debug_images
):

    fig, axes = plt.subplots(
        1,
        5,
        figsize=(
            15,
            3.5
        )
    )

    # --------------------------------------------------------
    # ORIGINAL
    # --------------------------------------------------------

    axes[0].imshow(
        original
    )

    axes[0].set_title(
        "Original"
    )

    axes[0].axis(
        "off"
    )

    # --------------------------------------------------------
    # GRAYSCALE
    # --------------------------------------------------------

    axes[1].imshow(
        debug_images[
            "Grayscale"
        ],
        cmap="gray"
    )

    axes[1].set_title(
        "Grayscale"
    )

    axes[1].axis(
        "off"
    )

    # --------------------------------------------------------
    # FOREGROUND
    # --------------------------------------------------------

    axes[2].imshow(
        debug_images[
            "Inverted / foreground"
        ],
        cmap="gray"
    )

    axes[2].set_title(
        "Foreground"
    )

    axes[2].axis(
        "off"
    )

    # --------------------------------------------------------
    # CROPPED
    # --------------------------------------------------------

    axes[3].imshow(
        debug_images[
            "Cropped"
        ],
        cmap="gray"
    )

    axes[3].set_title(
        "Cropped"
    )

    axes[3].axis(
        "off"
    )

    # --------------------------------------------------------
    # FINAL 28 × 28
    # --------------------------------------------------------

    axes[4].imshow(
        debug_images[
            "Centered 28×28"
        ],
        cmap="gray"
    )

    axes[4].set_title(
        "CNN Input"
    )

    axes[4].axis(
        "off"
    )

    fig.tight_layout()

    return fig
