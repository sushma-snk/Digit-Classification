import os
import pickle
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras.datasets import mnist


# ============================================================
# LOAD MNIST
# ============================================================

def load_mnist():

    print("Loading MNIST dataset...")

    (
        X_train,
        y_train
    ), (
        X_test,
        y_test
    ) = mnist.load_data()

    # --------------------------------------------------------
    # Normalize pixel values
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
    # Flatten images
    #
    # 28 × 28 → 784
    # --------------------------------------------------------

    X_train = X_train.reshape(
        X_train.shape[0],
        784
    )

    X_test = X_test.reshape(
        X_test.shape[0],
        784
    )

    return (
        X_train,
        y_train,
        X_test,
        y_test
    )


# ============================================================
# TRAIN MODEL
# ============================================================

def train_knn(
    X_train,
    y_train
):

    print(
        "Training KNN classifier..."
    )

    model = KNeighborsClassifier(
        n_neighbors=5,
        weights="distance",
        metric="euclidean",
        n_jobs=-1
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

    os.makedirs(
        os.path.dirname(path),
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

        model = pickle.load(
            file
        )

    return model


# ============================================================
# LOAD OR TRAIN
# ============================================================

def load_or_train_model(
    model_path="models/mnist_knn.pkl"
):

    # --------------------------------------------------------
    # Always load dataset
    # --------------------------------------------------------

    (
        X_train,
        y_train,
        X_test,
        y_test
    ) = load_mnist()


    # --------------------------------------------------------
    # Existing model
    # --------------------------------------------------------

    if os.path.exists(
        model_path
    ):

        print(
            "Loading saved KNN model..."
        )

        model = load_model(
            model_path
        )

    # --------------------------------------------------------
    # Train new model
    # --------------------------------------------------------

    else:

        print(
            "Saved model not found."
        )

        model = train_knn(
            X_train,
            y_train
        )

        save_model(
            model,
            model_path
        )

        print(
            f"Model saved to {model_path}"
        )


    return (
        model,
        X_train,
        y_train,
        X_test,
        y_test
    )


# ============================================================
# PREDICTION
# ============================================================

def predict_digit(
    model,
    features,
    k=5
):

    # --------------------------------------------------------
    # Temporarily change K
    # --------------------------------------------------------

    original_k = model.n_neighbors

    model.set_params(
        n_neighbors=k
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction = int(
        model.predict(
            features
        )[0]
    )

    # --------------------------------------------------------
    # Probability
    # --------------------------------------------------------

    probabilities = \
        model.predict_proba(
            features
        )[0]

    # --------------------------------------------------------
    # Restore original K
    # --------------------------------------------------------

    model.set_params(
        n_neighbors=original_k
    )

    return (
        prediction,
        probabilities
    )


# ============================================================
# FIND NEAREST NEIGHBORS
# ============================================================

def get_nearest_neighbors(
    model,
    features,
    X_train,
    y_train,
    k=5
):

    distances, indices = \
        model.kneighbors(
            features,
            n_neighbors=k
        )

    distances = distances[0]

    indices = indices[0]

    neighbor_images = X_train[
        indices
    ]

    neighbor_labels = y_train[
        indices
    ]

    return (
        distances,
        neighbor_images,
        neighbor_labels
    )
