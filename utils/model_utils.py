import tensorflow as tf


def load_model(path):

    """
    Load the trained MNIST CNN model.
    """

    model = tf.keras.models.load_model(
        path
    )

    return model


def predict_with_activations(
    model,
    x
):

    """
    Run prediction and also return the
    feature maps produced by convolution
    layers.

    Returns:

        probabilities
        activations
    """

    # ========================================================
    # FIND CONVOLUTION LAYERS
    # ========================================================

    convolution_layers = [

        layer

        for layer in model.layers

        if isinstance(
            layer,
            tf.keras.layers.Conv2D
        )

    ]

    # ========================================================
    # IF CONVOLUTION LAYERS EXIST
    # ========================================================

    if convolution_layers:

        outputs = [

            layer.output

            for layer in convolution_layers

        ]

        # ----------------------------------------------------
        # CREATE ACTIVATION MODEL
        # ----------------------------------------------------

        activation_model = tf.keras.Model(

            inputs=model.input,

            outputs=(
                outputs +
                [
                    model.output
                ]
            )
        )

        # ----------------------------------------------------
        # RUN MODEL
        # ----------------------------------------------------

        results = activation_model.predict(
            x,
            verbose=0
        )

        # ----------------------------------------------------
        # EXTRACT FEATURE MAPS
        # ----------------------------------------------------

        activations = {

            layer.name:
                result

            for layer, result

            in zip(
                convolution_layers,
                results[:-1]
            )

        }

        # ----------------------------------------------------
        # FINAL OUTPUT
        # ----------------------------------------------------

        probabilities = results[-1]

    else:

        # ====================================================
        # FALLBACK
        # ====================================================

        probabilities = model.predict(
            x,
            verbose=0
        )

        activations = {}

    return (
        probabilities,
        activations
    )
