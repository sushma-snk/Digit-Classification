import os
import pickle
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras.datasets import mnist


# ============================================================
# CONSTANTS
# ============================================================

DEFAULT_K = 5


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

    X_train = (
        X_train.astype(np.float32)
        / 255.0
    )

    X_test = (
        X_test.astype(np.float32)
        / 255.0
    )

    # 28 × 28 → 784
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
# TRAIN KNN
# ============================================================

def train_knn(
    X_train,
    y_train,
    k=DEFAULT_K
):

    model = KNeighborsClassifier(
        n_neighbors=k,
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

    directory = os.path.dirname(path)

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
# SAVE CUSTOM DATA
# ============================================================

def save_custom_data(
    X_custom,
    y_custom,
    path="models/custom_data.pkl"
):

    os.makedirs(
        os.path.dirname(path),
        exist_ok=True
    )

    data = {
        "X": X_custom,
        "y": y_custom
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
# LOAD CUSTOM DATA
# ============================================================

def load_custom_data(
    path="models/custom_data.pkl"
):

    if not os.path.exists(path):

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
# LOAD OR TRAIN MODEL
# ============================================================

def load_or_train_model(
    model_path="models/mnist_knn.pkl",
    custom_data_path="models/custom_data.pkl"
):

    (
        X_train,
        y_train,
        X_test,
        y_test
    ) = load_mnist()


    # --------------------------------------------------------
    # Load student-corrected examples
    # --------------------------------------------------------

    (
        X_custom,
        y_custom
    ) = load_custom_data(
        custom_data_path
    )


    # --------------------------------------------------------
    # Add custom examples
    # --------------------------------------------------------

    if len(X_custom) > 0:

        X_train_combined = np.vstack(
            [
                X_train,
                X_custom
            ]
        )

        y_train_combined = np.concatenate(
            [
                y_train,
                y_custom
            ]
        )

    else:

        X_train_combined = X_train

        y_train_combined = y_train


    # --------------------------------------------------------
    # Load existing model
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

        # Refit with custom examples if necessary
        if len(X_custom) > 0:

            model.fit(
                X_train_combined,
                y_train_combined
            )

    else:

        print(
            "Training new KNN model..."
        )

        model = train_knn(
            X_train_combined,
            y_train_combined
        )

        save_model(
            model,
            model_path
        )


    return (
        model,
        X_train_combined,
        y_train_combined,
        X_test,
        y_test,
        X_custom,
        y_custom
    )


# ============================================================
# ADD CORRECTED EXAMPLE
# ============================================================

def add_corrected_example(
    image_features,
    correct_label,
    X_custom,
    y_custom
):

    image_features = np.asarray(
        image_features,
        dtype=np.float32
    ).reshape(
        1,
        784
    )

    new_label = np.array(
        [correct_label],
        dtype=np.int64
    )

    if len(X_custom) == 0:

        X_updated = image_features

        y_updated = new_label

    else:

        X_updated = np.vstack(
            [
                X_custom,
                image_features
            ]
        )

        y_updated = np.concatenate(
            [
                y_custom,
                new_label
            ]
        )

    return (
        X_updated,
        y_updated
    )


# ============================================================
# RETRAIN / UPDATE MODEL
# ============================================================

def update_model(
    X_original,
    y_original,
    X_custom,
    y_custom,
    k=DEFAULT_K,
    model_path="models/mnist_knn.pkl"
):

    # --------------------------------------------------------
    # Combine original + corrected examples
    # --------------------------------------------------------

    if len(X_custom) > 0:

        X_combined = np.vstack(
            [
                X_original,
                X_custom
            ]
        )

        y_combined = np.concatenate(
            [
                y_original,
                y_custom
            ]
        )

    else:

        X_combined = X_original

        y_combined = y_original


    # --------------------------------------------------------
    # Train updated KNN
    # --------------------------------------------------------

    model = train_knn(
        X_combined,
        y_combined,
        k=k
    )


    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    save_model(
        model,
        model_path
    )


    return (
        model,
        X_combined,
        y_combined
    )


# ============================================================
# PREDICT
# ============================================================

def predict_digit(
    model,
    features,
    k=5
):

    original_k = model.n_neighbors

    model.set_params(
        n_neighbors=k
    )

    prediction = int(
        model.predict(
            features
        )[0]
    )

    probabilities = model.predict_proba(
        features
    )[0]

    model.set_params(
        n_neighbors=original_k
    )

    return (
        prediction,
        probabilities
    )


# ============================================================
# NEAREST NEIGHBORS
# ============================================================

def get_nearest_neighbors(
    model,
    features,
    X_train,
    y_train,
    k=5
):

    distances, indices = model.kneighbors(
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
