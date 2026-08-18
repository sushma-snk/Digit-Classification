import os
import numpy as np
import streamlit as st
from PIL import Image

from model_utils import (
    load_or_train_model,
    predict_digit,
    get_nearest_neighbors,
    add_corrected_example,
    update_model,
    save_custom_data
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
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .title {
        text-align: center;
        font-size: 42px;
        font-weight: bold;
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
    '<div class="title">'
    '🔢 MNIST Handwritten Digit Classifier'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning + Human-in-the-Loop Learning'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = (
    "models/mnist_knn.pkl"
)

CUSTOM_DATA_PATH = (
    "models/custom_data.pkl"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    return load_or_train_model(
        model_path=MODEL_PATH,
        custom_data_path=CUSTOM_DATA_PATH
    )


with st.spinner(
    "Loading MNIST classifier..."
):

    (
        model,
        X_train,
        y_train,
        X_test,
        y_test,
        X_custom,
        y_custom
    ) = load_model()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "⚙️ Controls"
)

show_preprocessing = st.sidebar.checkbox(
    "Show preprocessing",
    True
)

show_neighbors = st.sidebar.checkbox(
    "Show nearest neighbors",
    True
)

show_probabilities = st.sidebar.checkbox(
    "Show probabilities",
    True
)

k_value = st.sidebar.slider(
    "Number of neighbors (K)",
    1,
    9,
    5,
    2
)

st.sidebar.divider()

st.sidebar.subheader(
    "📚 Training Data"
)

st.sidebar.metric(
    "Original MNIST samples",
    "60,000"
)

st.sidebar.metric(
    "Student-corrected samples",
    len(X_custom)
)

st.sidebar.metric(
    "Total training samples",
    len(X_train)
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
    "Input method:",
    [
        "Upload image",
        "Use camera"
    ],
    horizontal=True
)

image = None


if input_method == "Upload image":

    uploaded_file = st.file_uploader(
        "Upload handwritten digit",
        type=[
            "png",
            "jpg",
            "jpeg"
        ]
    )

    if uploaded_file:

        image = Image.open(
            uploaded_file
        ).convert("L")

else:

    camera_image = st.camera_input(
        "Capture handwritten digit"
    )

    if camera_image:

        image = Image.open(
            camera_image
        ).convert("L")


# ============================================================
# PROCESS IMAGE
# ============================================================

if image is not None:

    # --------------------------------------------------------
    # Preprocess
    # --------------------------------------------------------

    processed_image, steps = \
        preprocess_image(
            image,
            return_steps=True
        )


    # --------------------------------------------------------
    # Display images
    # --------------------------------------------------------

    st.markdown(
        '<div class="section">'
        '2️⃣ Input'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "Original"
        )

        st.image(
            image,
            width=300
        )

    with col2:

        st.subheader(
            "MNIST-style"
        )

        st.image(
            processed_image,
            width=300
        )


    # ========================================================
    # PREPROCESSING
    # ========================================================

    if show_preprocessing:

        st.markdown(
            '<div class="section">'
            '3️⃣ Preprocessing'
            '</div>',
            unsafe_allow_html=True
        )

        cols = st.columns(4)

        names = [
            "Grayscale",
            "Normalized",
            "Digit extracted",
            "28 × 28"
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
    # FEATURES
    # ========================================================

    image_array = np.array(
        processed_image
    ).astype(
        np.float32
    )

    image_array /= 255.0

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
        '4️⃣ Prediction'
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
        Voting confidence:
        <b>{confidence * 100:.2f}%</b>
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    # ========================================================
    # PROBABILITIES
    # ========================================================

    if show_probabilities:

        st.markdown(
            '<div class="section">'
            '5️⃣ Class voting'
            '</div>',
            unsafe_allow_html=True
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
    # NEIGHBORS
    # ========================================================

    if show_neighbors:

        st.markdown(
            '<div class="section">'
            '6️⃣ Nearest training examples'
            '</div>',
            unsafe_allow_html=True
        )

        (
            distances,
            neighbor_images,
            neighbor_labels
        ) = get_nearest_neighbors(
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


    # ========================================================
    # HUMAN FEEDBACK
    # ========================================================

    st.markdown(
        '<div class="section">'
        '7️⃣ Was the prediction correct?'
        '</div>',
        unsafe_allow_html=True
    )

    feedback = st.radio(
        "Tell the system:",
        [
            "Yes, the prediction is correct",
            "No, the prediction is incorrect"
        ],
        key="prediction_feedback"
    )


    # ========================================================
    # CORRECT PREDICTION
    # ========================================================

    if feedback == \
            "Yes, the prediction is correct":

        st.success(
            f"Great! The model correctly classified "
            f"this digit as **{prediction}**."
        )


    # ========================================================
    # INCORRECT PREDICTION
    # ========================================================

    else:

        st.warning(
            f"The model predicted **{prediction}**. "
            "Let's correct it."
        )

        correct_label = st.selectbox(
            "What is the correct digit?",
            list(range(10)),
            key="correct_label"
        )


        # ----------------------------------------------------
        # Correct label button
        # ----------------------------------------------------

        if st.button(
            "➕ Add corrected example and update model",
            type="primary"
        ):

            # -----------------------------------------------
            # Add example
            # -----------------------------------------------

            (
                X_custom_updated,
                y_custom_updated
            ) = add_corrected_example(
                features,
                correct_label,
                X_custom,
                y_custom
            )


            # -----------------------------------------------
            # Save custom data
            # -----------------------------------------------

            save_custom_data(
                X_custom_updated,
                y_custom_updated,
                CUSTOM_DATA_PATH
            )


            # -----------------------------------------------
            # Update KNN
            # -----------------------------------------------

            (
                updated_model,
                X_updated,
                y_updated
            ) = update_model(
                X_train,
                y_train,
                X_custom_updated,
                y_custom_updated,
                k=k_value,
                model_path=MODEL_PATH
            )


            # -----------------------------------------------
            # Update Streamlit state
            # -----------------------------------------------

            st.session_state[
                "model_updated"
            ] = True

            st.session_state[
                "updated_model"
            ] = updated_model

            st.session_state[
                "updated_X"
            ] = X_updated

            st.session_state[
                "updated_y"
            ] = y_updated

            st.session_state[
                "correct_label"
            ] = correct_label


            st.success(
                f"""
                Example added successfully!

                **Original prediction:** {prediction}

                **Correct label:** {correct_label}

                **New training size:** {len(X_updated):,}
                """
            )


            st.rerun()


    # ========================================================
    # UPDATED MODEL
    # ========================================================

    if st.session_state.get(
        "model_updated",
        False
    ):

        updated_model = \
            st.session_state[
                "updated_model"
            ]

        updated_X = \
            st.session_state[
                "updated_X"
            ]

        updated_y = \
            st.session_state[
                "updated_y"
            ]

        correct_label = \
            st.session_state[
                "correct_label"
            ]


        st.markdown(
            '<div class="section">'
            '8️⃣ Updated model'
            '</div>',
            unsafe_allow_html=True
        )

        st.info(
            f"""
            The corrected image has now been added to the
            training dataset.

            **Original prediction:** {prediction}

            **Correct label:** {correct_label}

            **Training examples before correction:**
            {len(X_train):,}

            **Training examples after correction:**
            {len(updated_X):,}
            """
        )


        # ----------------------------------------------------
        # Predict again
        # ----------------------------------------------------

        new_prediction, new_probabilities = \
            predict_digit(
                updated_model,
                features,
                k=k_value
            )


        st.subheader(
            "Prediction after learning"
        )

        if new_prediction == correct_label:

            st.success(
                f"""
                🎉 The updated classifier now predicts:

                **{new_prediction}**

                which matches the corrected label.
                """
            )

        else:

            st.warning(
                f"""
                The updated classifier predicts:

                **{new_prediction}**

                The corrected label was:

                **{correct_label}**

                This is actually useful for demonstrating
                that adding one example does not necessarily
                guarantee a change in the classifier's decision.
                """
            )


        # ----------------------------------------------------
        # Updated probabilities
        # ----------------------------------------------------

        fig = plot_probabilities(
            new_probabilities,
            new_prediction
        )

        st.pyplot(
            fig,
            use_container_width=True
        )


    # ========================================================
    # EXPLANATION
    # ========================================================

    st.markdown(
        '<div class="section">'
        '🧠 What just happened?'
        '</div>',
        unsafe_allow_html=True
    )

    st.info(
        """
        This demonstration uses **human-in-the-loop learning**.

        When the model makes a mistake:

        **1. Human identifies the mistake**

        ↓

        **2. Human provides the correct label**

        ↓

        **3. The corrected image becomes a new labelled
        training example**

        ↓

        **4. The KNN classifier is updated**

        ↓

        **5. Future predictions can use this new example**

        This is a simple example of how machine-learning
        systems can incorporate new labelled data.
        """
    )


# ============================================================
# INITIAL MESSAGE
# ============================================================

else:

    st.info(
        "👆 Upload or capture a handwritten digit to begin."
    )

    st.markdown(
        """
        ### 🎓 Demonstration flow

        ```text
        Handwritten Digit
                ↓
        Image Preprocessing
                ↓
        28 × 28 pixels
                ↓
        784 Features
                ↓
        KNN
                ↓
        Nearest Neighbors
                ↓
        Voting
                ↓
        Prediction
                ↓
        Human Feedback
                ↓
        Correct Label
                ↓
        Updated Training Data
        ```
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "MNIST Machine Learning Demonstration | "
    "KNN + Human-in-the-Loop Learning"
)
