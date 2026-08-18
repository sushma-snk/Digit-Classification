import os
import pickle
import numpy as np

from sklearn.svm import SVC
from tensorflow.keras.datasets import mnist


# ============================================================
# SETTINGS
# ============================================================

DEFAULT_TRAINING_SAMPLES = 20000

SVM_C = 10

SVM_GAMMA = "scale"


# ============================================================
# LOAD MNIST
# ============================================================

def load_mnist():

    print("Loading MNIST...")

    (
        X_train,
        y_train
    ), (
        X_test,
        y_test
    ) = mnist.load_data()


    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    X_train = (
        X_train.astype(
            np.float32
        ) / 255.0
    )

    X_test = (
        X_test.astype(
            np.float32
        ) / 255.0
    )


    # --------------------------------------------------------
    # Flatten
    # --------------------------------------------------------

    X_train = X_train.reshape(
        X_train.shape[0],
        784
    )

    X_test = X_test.reshape(
        X_test.shape[0],
        784
    )


    # --------------------------------------------------------
    # Use a subset for faster SVM training
    # --------------------------------------------------------

    rng = np.random.default_rng(
        42
    )

    indices = rng.choice(
        len(X_train),
        size=DEFAULT_TRAINING_SAMPLES,
        replace=False
    )

    X_train = X_train[
        indices
    ]

    y_train = y_train[
        indices
    ]


    return (
        X_train,
        y_train,
        X_test,
        y_test
    )


# ============================================================
# TRAIN SVM
# ============================================================

def train_svm(
    X_train,
    y_train
):

    print(
        "Training SVM..."
    )

    model = SVC(
        kernel="rbf",
        C=SVM_C,
        gamma=SVM_GAMMA,
        probability=True,
        random_state=42,
        cache_size=1000
    )

    model.fit(
        X_train,
        y_train
    )

    return model


# ============================================================
# SAVE MODEL
# ============================================================

def save_model(
    model,
    path
):

    directory = os.path.dirname(
        path
    )

    if directory:

        os.makedirs(
            directory,
            exist_ok=True
        )

    with open(
        path,
        "wb"
    ) as file:

        pickle.dump(
            model,
            file
        )


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(
    path
):

    with open(
        path,
        "rb"
    ) as file:

        return pickle.load(
            file
        )


# ============================================================
# CUSTOM DATA
# ============================================================

def load_custom_data(
    path
):

    if not os.path.exists(
        path
    ):

        return (
            np.empty(
                (0, 784),
                dtype=np.float32
            ),
            np.empty(
                (0,),
                dtype=np.int64
            )
        )


    with open(
        path,
        "rb"
    ) as file:

        data = pickle.load(
            file
        )


    return (
        data["X"],
        data["y"]
    )


# ============================================================
# SAVE CUSTOM DATA
# ============================================================

def save_custom_data(
    X,
    y,
    path
):

    directory = os.path.dirname(
        path
    )

    if directory:

        os.makedirs(
            directory,
            exist_ok=True
        )

    data = {
        "X": X,
        "y": y
    }


    with open(
        path,
        "wb"
    ) as file:

        pickle.dump(
            data,
            file
        )


# ============================================================
# LOAD OR TRAIN MODEL
# ============================================================

def load_or_train_model(
    model_path,
    custom_data_path
):

    (
        X_train,
        y_train,
        X_test,
        y_test
    ) = load_mnist()


    # --------------------------------------------------------
    # Load custom examples
    # --------------------------------------------------------

    (
        X_custom,
        y_custom
    ) = load_custom_data(
        custom_data_path
    )


    # --------------------------------------------------------
    # Combine data
    # --------------------------------------------------------

    if len(X_custom) > 0:

        X_combined = np.vstack(
            [
                X_train,
                X_custom
            ]
        )

        y_combined = np.concatenate(
            [
                y_train,
                y_custom
            ]
        )

    else:

        X_combined = X_train

        y_combined = y_train


    # --------------------------------------------------------
    # Existing model
    # --------------------------------------------------------

    if os.path.exists(
        model_path
    ):

        print(
            "Loading saved SVM..."
        )

        model = load_model(
            model_path
        )


        # If custom examples exist,
        # retrain using the combined data.

        if len(X_custom) > 0:

            model.fit(
                X_combined,
                y_combined
            )

            save_model(
                model,
                model_path
            )


    # --------------------------------------------------------
    # New model
    # --------------------------------------------------------

    else:

        model = train_svm(
            X_combined,
            y_combined
        )

        save_model(
            model,
            model_path
        )


    return (
        model,
        X_combined,
        y_combined,
        X_test,
        y_test,
        X_custom,
        y_custom
    )


# ============================================================
# PREDICT
# ============================================================

def predict_digit(
    model,
    features
):

    prediction = int(
        model.predict(
            features
        )[0]
    )


    probabilities = \
        model.predict_proba(
            features
        )[0]


    return (
        prediction,
        probabilities
    )


# ============================================================
# RETRAIN WITH NEW EXAMPLE
# ============================================================

def retrain_with_new_example(
    features,
    correct_label,
    X_train,
    y_train,
    X_custom,
    y_custom,
    model_path,
    custom_data_path
):

    # --------------------------------------------------------
    # Convert label
    # --------------------------------------------------------

    correct_label = int(
        correct_label
    )


    # --------------------------------------------------------
    # Add new example to custom data
    # --------------------------------------------------------

    new_X = features.astype(
        np.float32
    )

    new_y = np.array(
        [correct_label],
        dtype=np.int64
    )


    if len(X_custom) == 0:

        updated_X_custom = new_X

        updated_y_custom = new_y

    else:

        updated_X_custom = np.vstack(
            [
                X_custom,
                new_X
            ]
        )

        updated_y_custom = np.concatenate(
            [
                y_custom,
                new_y
            ]
        )


    # --------------------------------------------------------
    # Save custom examples
    # --------------------------------------------------------

    save_custom_data(
        updated_X_custom,
        updated_y_custom,
        custom_data_path
    )


    # --------------------------------------------------------
    # Combine with original training data
    # --------------------------------------------------------

    X_combined = np.vstack(
        [
            X_train,
            new_X
        ]
    )

    y_combined = np.concatenate(
        [
            y_train,
            new_y
        ]
    )


    # --------------------------------------------------------
    # Retrain SVM
    # --------------------------------------------------------

    new_model = train_svm(
        X_combined,
        y_combined
    )


    # --------------------------------------------------------
    # Save updated model
    # --------------------------------------------------------

    save_model(
        new_model,
        model_path
    )


    return (
        new_model,
        X_combined,
        y_combined,
        updated_X_custom,
        updated_y_custom
    )
