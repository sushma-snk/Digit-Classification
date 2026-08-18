import os
import numpy as np
import streamlit as st
from PIL import Image

from model_utils import (
    load_or_train_model,
    predict_digit,
    get_nearest_neighbors
)

from preprocessing import preprocess_image

from visualization import (
    plot_probabilities,
    plot_nearest_neighbors
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="MNIST ML Classifier",
    page_icon="🔢",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .title {
        text-align: center;
        font-size: 42px;
        font-weight: bold;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #666;
        font-size: 18px;
        margin-bottom: 30px;
    }

    .prediction-box {
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        background-color: #f1f5f9;
    }

    .prediction {
        font-size: 80px;
        font-weight: bold;
    }

    .section {
        font-size: 26px;
        font-weight: bold;
        margin-top: 25px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="title">🔢 MNIST Handwritten Digit Classifier</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Interactive Machine Learning Demonstration using K-Nearest Neighbors'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# MODEL
# ============================================================

MODEL_PATH = "models/mnist_knn.pkl"


@st.cache_resource
def load_model():

    model, X_train, y_train, X_test, y_test = \
        load_or_train_model(
            model_path=MODEL_PATH
        )

    return (
        model,
        X_train,
        y_train,
        X_test,
        y_test
    )


with st.spinner(
    "Loading MNIST dataset and KNN model..."
):

    model, X_train, y_train, X_test, y_test = load_model()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "⚙️ Demonstration Controls"
)

show_preprocessing = st.sidebar.checkbox(
    "Show preprocessing",
    value=True
)

show_neighbors = st.sidebar.checkbox(
    "Show nearest neighbors",
    value=True
)

show_probabilities = st.sidebar.checkbox(
    "Show class probabilities",
    value=True
)

k_value = st.sidebar.slider(
    "Number of neighbors (K)",
    min_value=1,
    max_value=9,
    value=5,
    step=2
)

st.sidebar.divider()

st.sidebar.write(
    """
    ### Algorithm

    **K-Nearest Neighbors (KNN)**

    The classifier compares the input digit with
    stored MNIST training examples.

    The K most similar examples vote for the
    final class.
    """
)


# ============================================================
# INPUT
# ============================================================

st.markdown(
    '<div class="section">'
    '1️⃣ Provide a handwritten digit'
    '</div>',
    unsafe_allow_html=True
)

input_method = st.radio(
    "Choose input method:",
    [
        "Upload image",
        "Use camera"
    ],
    horizontal=True
)

image = None


# ============================================================
# UPLOAD
# ============================================================

if input_method == "Upload image":

    uploaded_file = st.file_uploader(
        "Upload a handwritten digit",
        type=[
            "png",
            "jpg",
            "jpeg"
        ]
    )

    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        ).convert("L")


# ============================================================
# CAMERA
# ============================================================

else:

    camera_image = st.camera_input(
        "Take a photograph of a handwritten digit"
    )

    if camera_image is not None:

        image = Image.open(
            camera_image
        ).convert("L")


# ============================================================
# CLASSIFICATION
# ============================================================

if image is not None:

    # --------------------------------------------------------
    # IMAGE DISPLAY
    # --------------------------------------------------------

    st.markdown(
        '<div class="section">'
        '2️⃣ Input image'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "Original image"
        )

        st.image(
            image,
            width=300
        )


    # --------------------------------------------------------
    # PREPROCESSING
    # --------------------------------------------------------

    processed_image, steps = \
        preprocess_image(
            image,
            return_steps=True
        )

    with col2:

        st.subheader(
            "MNIST-style image"
        )

        st.image(
            processed_image,
            width=300
        )


    # ========================================================
    # PREPROCESSING VISUALIZATION
    # ========================================================

    if show_preprocessing:

        st.markdown(
            '<div class="section">'
            '3️⃣ Preprocessing'
            '</div>',
            unsafe_allow_html=True
        )

        st.write(
            """
            The original photograph cannot be directly given
            to the MNIST classifier. It is converted into the
            same 28 × 28 grayscale representation used by MNIST.
            """
        )

        cols = st.columns(4)

        names = [
            "Grayscale",
            "Normalized",
            "Digit extracted",
            "28 × 28 image"
        ]

        for i in range(4):

            with cols[i]:

                st.caption(
                    names[i]
                )

                st.image(
                    steps[i],
                    use_container_width=True
                )


    # ========================================================
    # CONVERT TO 784 FEATURES
    # ========================================================

    image_array = np.array(
        processed_image
    ).astype(
        np.float32
    )

    # Normalize
    image_array /= 255.0

    # 28 × 28 → 784
    features = image_array.reshape(
        1,
        784
    )


    # ========================================================
    # PREDICTION
    # ========================================================

    prediction, probabilities = \
        predict_digit(
            model,
            features,
            k=k_value
        )

    confidence = probabilities[
        prediction
    ]


    # ========================================================
    # RESULT
    # ========================================================

    st.markdown(
        '<div class="section">'
        '4️⃣ Classification result'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="prediction-box">

        <div>Predicted digit</div>

        <div class="prediction">
        {prediction}
        </div>

        <div>
        Estimated confidence:
        <b>{confidence * 100:.2f}%</b>
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    # ========================================================
    # PROBABILITY DISTRIBUTION
    # ========================================================

    if show_probabilities:

        st.markdown(
            '<div class="section">'
            '5️⃣ Class probabilities'
            '</div>',
            unsafe_allow_html=True
        )

        st.write(
            """
            KNN does not naturally produce probabilities like
            a neural network. Here the values are calculated
            from the voting distribution of the K nearest
            neighbors.
            """
        )

        fig = plot_probabilities(
            probabilities,
            prediction
        )

        st.pyplot(
            fig,
            use_container_width=True
        )


    # ========================================================
    # NEAREST NEIGHBORS
    # ========================================================

    if show_neighbors:

        st.markdown(
            '<div class="section">'
            '6️⃣ Which training examples influenced the decision?'
            '</div>',
            unsafe_allow_html=True
        )

        st.write(
            f"""
            KNN finds the **{k_value} most similar images**
            in the MNIST training dataset.

            These examples are the neighbors used to determine
            the predicted class.
            """
        )

        distances, neighbor_images, neighbor_labels = \
            get_nearest_neighbors(
                model,
                features,
                X_train,
                y_train,
                k=k_value
            )

        fig = plot_nearest_neighbors(
            neighbor_images,
            neighbor_labels,
            distances
        )

        st.pyplot(
            fig,
            use_container_width=True
        )


        # ----------------------------------------------------
        # VOTING TABLE
        # ----------------------------------------------------

        st.subheader(
            "Neighbor voting"
        )

        vote_counts = {}

        for label in neighbor_labels:

            label = int(label)

            vote_counts[label] = \
                vote_counts.get(
                    label,
                    0
                ) + 1

        sorted_votes = sorted(
            vote_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )

        for label, count in sorted_votes:

            percentage = (
                count /
                k_value
            ) * 100

            st.write(
                f"**Digit {label}:** "
                f"{count}/{k_value} votes "
                f"({percentage:.1f}%)"
            )


    # ========================================================
    # HOW KNN WORKS
    # ========================================================

    st.markdown(
        '<div class="section">'
        '🧠 How did KNN classify the digit?'
        '</div>',
        unsafe_allow_html=True
    )

    st.info(
        f"""
        **Step 1 — Image representation**

        The 28 × 28 image contains 784 pixels.
        Each pixel becomes a feature.

        **Step 2 — Distance calculation**

        KNN compares the uploaded digit with training
        examples using distance between their 784-dimensional
        feature vectors.

        **Step 3 — Find nearest neighbors**

        The algorithm selects the {k_value} most similar
        training images.

        **Step 4 — Voting**

        The neighboring images vote for their digit classes.

        **Step 5 — Final prediction**

        The digit receiving the strongest vote is selected.

        **Final prediction: {prediction}**
        """
    )


# ============================================================
# INITIAL SCREEN
# ============================================================

else:

    st.info(
        "👆 Upload a handwritten digit or use the camera to begin."
    )

    st.markdown(
        """
        ### ✍️ For the best results

        Write one digit:

        - Use a black or dark pen.
        - Use white paper.
        - Make the digit reasonably large.
        - Keep it approximately centered.
        - Avoid multiple digits in the same image.

        ### 🎓 What students can learn

        This application demonstrates:

        **Image → Features → Distance → Neighbors → Voting → Classification**
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "MNIST Machine Learning Demonstration • "
    "K-Nearest Neighbors + Streamlit"
)
