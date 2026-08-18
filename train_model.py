import os
import tensorflow as tf


# ============================================================
# MODEL PATH
# ============================================================

os.makedirs(
    "models",
    exist_ok=True
)

MODEL_PATH = (
    "models/mnist_cnn.keras"
)


# ============================================================
# LOAD MNIST
# ============================================================

print(
    "Downloading/loading MNIST..."
)

(
    x_train,
    y_train
), (
    x_test,
    y_test
) = tf.keras.datasets.mnist.load_data()


# ============================================================
# NORMALIZE
# ============================================================

x_train = (
    x_train.astype(
        "float32"
    ) / 255.0
)

x_test = (
    x_test.astype(
        "float32"
    ) / 255.0
)


# ============================================================
# ADD CHANNEL
# ============================================================

x_train = x_train[
    ...,
    None
]

x_test = x_test[
    ...,
    None
]


# ============================================================
# CNN
# ============================================================

model = tf.keras.Sequential([

    tf.keras.layers.Input(
        shape=(
            28,
            28,
            1
        )
    ),

    tf.keras.layers.Conv2D(
        32,
        (
            3,
            3
        ),
        activation="relu",
        padding="same",
        name="conv1"
    ),

    tf.keras.layers.MaxPooling2D(
        (
            2,
            2
        ),
        name="pool1"
    ),

    tf.keras.layers.Conv2D(
        64,
        (
            3,
            3
        ),
        activation="relu",
        padding="same",
        name="conv2"
    ),

    tf.keras.layers.MaxPooling2D(
        (
            2,
            2
        ),
        name="pool2"
    ),

    tf.keras.layers.Flatten(
        name="flatten"
    ),

    tf.keras.layers.Dense(
        128,
        activation="relu",
        name="dense1"
    ),

    tf.keras.layers.Dropout(
        0.3,
        name="dropout"
    ),

    tf.keras.layers.Dense(
        10,
        activation="softmax",
        name="output"
    )

])


# ============================================================
# COMPILE
# ============================================================

model.compile(

    optimizer="adam",

    loss=(
        "sparse_categorical_crossentropy"
    ),

    metrics=[
        "accuracy"
    ]

)


# ============================================================
# TRAIN
# ============================================================

print(
    "Training CNN..."
)

model.fit(

    x_train,

    y_train,

    epochs=5,

    batch_size=128,

    validation_split=0.1,

    verbose=1

)


# ============================================================
# EVALUATION
# ============================================================

print(
    "Evaluating..."
)

loss, accuracy = model.evaluate(

    x_test,

    y_test,

    verbose=0

)


print(
    f"Test accuracy: {accuracy:.4f}"
)


# ============================================================
# SAVE
# ============================================================

model.save(
    MODEL_PATH
)

print(
    "Model saved:"
)

print(
    MODEL_PATH
)
