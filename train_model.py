import tensorflow as tf


def build_model():

    model = tf.keras.Sequential(
        [

            tf.keras.layers.Input(
                shape=(28, 28, 1)
            ),

            # --------------------------------------------
            # First convolution block
            # --------------------------------------------

            tf.keras.layers.Conv2D(
                32,
                kernel_size=(3, 3),
                activation="relu",
                name="conv1"
            ),

            tf.keras.layers.MaxPooling2D(
                pool_size=(2, 2),
                name="pool1"
            ),

            # --------------------------------------------
            # Second convolution block
            # --------------------------------------------

            tf.keras.layers.Conv2D(
                64,
                kernel_size=(3, 3),
                activation="relu",
                name="conv2"
            ),

            tf.keras.layers.MaxPooling2D(
                pool_size=(2, 2),
                name="pool2"
            ),

            # --------------------------------------------
            # Feature representation
            # --------------------------------------------

            tf.keras.layers.Flatten(
                name="flatten"
            ),

            tf.keras.layers.Dense(
                128,
                activation="relu",
                name="dense_features"
            ),

            tf.keras.layers.Dropout(
                0.3
            ),

            # --------------------------------------------
            # Output layer
            # --------------------------------------------

            tf.keras.layers.Dense(
                10,
                activation="softmax",
                name="output"
            )
        ]
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def train_model(
    epochs=3,
    save_path="mnist_cnn.keras"
):

    print("Loading MNIST dataset...")

    (x_train, y_train), (x_test, y_test) = (
        tf.keras.datasets.mnist.load_data()
    )

    # --------------------------------------------
    # Normalize
    # --------------------------------------------

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # --------------------------------------------
    # Add channel dimension
    # --------------------------------------------

    x_train = x_train[..., np.newaxis]
    x_test = x_test[..., np.newaxis]

    # --------------------------------------------
    # Build model
    # --------------------------------------------

    model = build_model()

    print(
        "\nTraining CNN..."
    )

    model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=epochs,
        batch_size=128,
        verbose=1
    )

    # --------------------------------------------
    # Evaluate
    # --------------------------------------------

    test_loss, test_accuracy = model.evaluate(
        x_test,
        y_test,
        verbose=0
    )

    print(
        f"\nTest accuracy: "
        f"{test_accuracy * 100:.2f}%"
    )

    # --------------------------------------------
    # Save
    # --------------------------------------------

    model.save(
        save_path
    )

    print(
        f"Model saved to: {save_path}"
    )

    return model


if __name__ == "__main__":

    import numpy as np

    train_model(
        epochs=3,
        save_path="mnist_cnn.keras"
    )
