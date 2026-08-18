import numpy as np
from PIL import Image, ImageOps, ImageFilter


def preprocess_image(
    image,
    return_steps=False
):
    """
    Convert a real-world handwritten digit image
    into MNIST-like 28 x 28 format.

    Returns
    -------
    processed_image : PIL.Image
        Final 28 x 28 image.

    processing_steps : list
        Intermediate processing images.
    """

    # ========================================================
    # STEP 1: Grayscale
    # ========================================================

    gray = image.convert("L")

    # Slight smoothing
    gray = gray.filter(
        ImageFilter.GaussianBlur(radius=0.5)
    )

    step1 = gray.copy()


    # ========================================================
    # Convert to numpy
    # ========================================================

    arr = np.array(gray).astype(
        np.float32
    )


    # ========================================================
    # STEP 2: Determine foreground/background
    # ========================================================

    h, w = arr.shape

    border_pixels = np.concatenate(
        [
            arr[0, :],
            arr[-1, :],
            arr[:, 0],
            arr[:, -1]
        ]
    )

    border_mean = np.mean(
        border_pixels
    )

    center_region = arr[
        h // 4 : 3 * h // 4,
        w // 4 : 3 * w // 4
    ]

    center_mean = np.mean(
        center_region
    )

    # MNIST format:
    # background = black
    # digit = white

    if center_mean < border_mean:

        arr = 255.0 - arr


    # ========================================================
    # STEP 3: Normalize
    # ========================================================

    min_value = arr.min()
    max_value = arr.max()

    if max_value > min_value:

        arr = (
            (arr - min_value)
            /
            (max_value - min_value)
            * 255.0
        )

    normalized = Image.fromarray(
        arr.astype(np.uint8)
    )

    step2 = normalized.copy()


    # ========================================================
    # STEP 4: Threshold
    # ========================================================

    threshold = np.percentile(
        arr,
        65
    )

    binary = np.where(
        arr > threshold,
        255,
        0
    ).astype(np.uint8)

    binary_image = Image.fromarray(
        binary
    )

    # ========================================================
    # Find digit bounding box
    # ========================================================

    bbox = binary_image.getbbox()

    if bbox is not None:

        left, top, right, bottom = bbox

        # Add margin
        margin = int(
            0.15 *
            max(
                right - left,
                bottom - top
            )
        )

        left = max(
            0,
            left - margin
        )

        top = max(
            0,
            top - margin
        )

        right = min(
            binary_image.width,
            right + margin
        )

        bottom = min(
            binary_image.height,
            bottom + margin
        )

        cropped = binary_image.crop(
            (
                left,
                top,
                right,
                bottom
            )
        )

    else:

        cropped = binary_image


    step3 = cropped.copy()


    # ========================================================
    # STEP 5: Resize while preserving aspect ratio
    # ========================================================

    target_size = 20

    width, height = cropped.size

    if width > height:

        new_width = target_size

        new_height = max(
            1,
            int(
                height *
                target_size /
                width
            )
        )

    else:

        new_height = target_size

        new_width = max(
            1,
            int(
                width *
                target_size /
                height
            )
        )

    resized = cropped.resize(
        (
            new_width,
            new_height
        ),
        Image.Resampling.LANCZOS
    )


    # ========================================================
    # STEP 6: Put digit in 28 x 28 canvas
    # ========================================================

    canvas = Image.new(
        "L",
        (28, 28),
        0
    )

    x_offset = (
        28 - resized.width
    ) // 2

    y_offset = (
        28 - resized.height
    ) // 2

    canvas.paste(
        resized,
        (
            x_offset,
            y_offset
        )
    )

    processed = canvas


    # ========================================================
    # Return
    # ========================================================

    if return_steps:

        return (
            processed,
            [
                step1,
                step2,
                step3,
                processed
            ]
        )

    return processed
