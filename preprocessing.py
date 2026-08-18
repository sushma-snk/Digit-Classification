import numpy as np

from PIL import (
    Image,
    ImageFilter
)


def preprocess_image(
    image,
    return_steps=False
):

    # ========================================================
    # 1. GRAYSCALE
    # ========================================================

    gray = image.convert(
        "L"
    )

    gray = gray.filter(
        ImageFilter.GaussianBlur(
            radius=0.5
        )
    )

    step1 = gray.copy()


    # ========================================================
    # 2. CONVERT TO ARRAY
    # ========================================================

    arr = np.array(
        gray
    ).astype(
        np.float32
    )


    height, width = arr.shape


    # ========================================================
    # 3. DETECT BACKGROUND
    # ========================================================

    border = np.concatenate(
        [
            arr[0, :],
            arr[-1, :],
            arr[:, 0],
            arr[:, -1]
        ]
    )

    border_mean = np.mean(
        border
    )


    center = arr[
        height // 4:
        3 * height // 4,

        width // 4:
        3 * width // 4
    ]

    center_mean = np.mean(
        center
    )


    # MNIST:
    # black background
    # white digit

    if center_mean < border_mean:

        arr = (
            255.0 -
            arr
        )


    # ========================================================
    # 4. NORMALIZE
    # ========================================================

    minimum = arr.min()

    maximum = arr.max()


    if maximum > minimum:

        arr = (
            (arr - minimum)
            /
            (maximum - minimum)
            * 255.0
        )


    normalized = Image.fromarray(
        arr.astype(
            np.uint8
        )
    )

    step2 = normalized.copy()


    # ========================================================
    # 5. THRESHOLD
    # ========================================================

    threshold = np.percentile(
        arr,
        65
    )


    binary = np.where(
        arr > threshold,
        255,
        0
    ).astype(
        np.uint8
    )


    binary_image = Image.fromarray(
        binary
    )


    # ========================================================
    # 6. FIND DIGIT
    # ========================================================

    bbox = binary_image.getbbox()


    if bbox is not None:

        left, top, right, bottom = bbox


        digit_width = (
            right - left
        )

        digit_height = (
            bottom - top
        )


        margin = int(
            0.15 *
            max(
                digit_width,
                digit_height
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
            width,
            right + margin
        )

        bottom = min(
            height,
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
    # 7. RESIZE
    # ========================================================

    target_size = 20


    crop_width, crop_height = \
        cropped.size


    if crop_width >= crop_height:

        new_width = target_size

        new_height = max(
            1,
            int(
                crop_height *
                target_size /
                crop_width
            )
        )

    else:

        new_height = target_size

        new_width = max(
            1,
            int(
                crop_width *
                target_size /
                crop_height
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
    # 8. 28 × 28 CANVAS
    # ========================================================

    canvas = Image.new(
        "L",
        (
            28,
            28
        ),
        0
    )


    x = (
        28 -
        resized.width
    ) // 2


    y = (
        28 -
        resized.height
    ) // 2


    canvas.paste(
        resized,
        (
            x,
            y
        )
    )


    processed = canvas


    # ========================================================
    # RETURN
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
