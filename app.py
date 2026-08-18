import os
import numpy as np
import streamlit as st
from PIL import Image

from model_utils import (
    load_or_train_model,
    predict_digit,
    retrain_with_new_example,
    save_custom_data
)

from preprocessing import preprocess_image

from visualization import (
    plot_probabilities,
    plot_feature_importance,
    plot_confusion_matrix
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="MNIST SVM Digit Classifier",
    page_icon="🔢",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #666;
        margin-bottom: 30px;
    }

    .section-title {
        font-size: 27px;
        font-weight: 700;
        margin-top: 25px;
        margin-bottom: 15px;
    }

    .prediction-box {
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        background-color: #f1f5f9;
        margin-top: 15px;
    }

    .prediction-digit {
        font-size: 80px;
        font-weight: 800;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">'
    '🔢 MNIST Handwritten Digit Classifier'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Interactive Machine Learning Demonstration using Support Vector Machine'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# FILE PATHS
# ============================================================

MODEL_PATH = "models/mnist_svm.pkl"

CUSTOM_DATA_PATH = "models/custom_data.pkl"


# ============================================================
# INITIALIZE SESSION STATE
# ============================================================

if "model_version" not in st.session_state:

    st.session_state.model_version = 0


if "correction_message" not in st.session_state:

    st.session_state.correction_message = None


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model(version=0):

    return load_or_train_model(
        model_path=MODEL_PATH,
        custom_data_path=CUSTOM_DATA_PATH
    )


with st.spinner(
    "Loading SVM classifier..."
):

    (
        model,
        X_train,
        y_train,
        X_test,
        y_test,
        X_custom,
        y_custom
    ) = load_model(
        st.session_state.model_version
    )


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

show_probabilities = st.sidebar.checkbox(
    "Show class probabilities",
    value=True
)

show_features = st.sidebar.checkbox(
    "Show pixel feature importance",
    value=True
)

st.sidebar.divider()

st.sidebar.subheader(
    "📊 Training Information"
)

st.sidebar.metric(
    "Original MNIST samples",
    "60,000"
)

st.sidebar.metric(
    "Student-added samples",
    len(X_custom)
)

st.sidebar.metric(
    "Current training samples",
    len(X_train)
)

st.sidebar.divider()

st.sidebar.write(
    """
    ### SVM

    The Support Vector Machine finds a decision boundary
    that separates the different digit classes.

    For this demonstration, an RBF kernel is used.
    """
)


# ============================================================
# MAIN INPUT SECTION
# ============================================================

st.markdown(
    '<div class="section-title">'
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
        "Capture a handwritten digit"
    )

    if camera_image is not None:

        image = Image.open(
            camera_image
        ).convert("L")


# ============================================================
# CLASSIFICATION
# ============================================================

if image is not None:

    # ========================================================
    # DISPLAY INPUT
    # ========================================================

    st.markdown(
        '<div class="section-title">'
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


    # ========================================================
    # PREPROCESS
    # ========================================================

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
            '<div class="section-title">'
            '3️⃣ Image preprocessing'
            '</div>',
            unsafe_allow_html=True
        )

        st.write(
            """
            The SVM was trained using 28 × 28 MNIST images.
            Therefore, the camera/uploaded image must first be
            converted into the same representation.
            """
        )

        columns = st.columns(4)

        names = [
            "Grayscale",
            "Normalized",
            "Digit extracted",
            "28 × 28"
        ]

        for i in range(4):

            with columns[i]:

                st.caption(
                    names[i]
                )

                st.image(
                    steps[i],
                    use_container_width=True
                )


    # ========================================================
    # CREATE 784 FEATURES
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
            features
        )

    confidence = float(
        probabilities[prediction]
    )


    # ========================================================
    # RESULT
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '4️⃣ SVM classification result'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="prediction-box">

        <div>Predicted digit</div>

        <div class="prediction-digit">
        {prediction}
        </div>

        <div>
        SVM confidence:
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
            '<div class="section-title">'
            '5️⃣ SVM class probabilities'
            '</div>',
            unsafe_allow_html=True
        )

        st.write(
            """
            The SVM produces a score for each digit class.
            These scores are converted into probability-like
            values using probability calibration.
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
    # FEATURE IMPORTANCE
    # ========================================================

    if show_features:

        st.markdown(
            '<div class="section-title">'
            '6️⃣ What pixels are important?'
            '</div>',
            unsafe_allow_html=True
        )

        st.write(
            """
            Each MNIST image contains 784 pixel features.
            The visualization below shows the influence of
            the SVM coefficients for the predicted digit.
            """
        )

        if hasattr(
            model,
            "coef_"
        ):

            fig = plot_feature_importance(
                model,
                prediction
            )

            st.pyplot(
                fig,
                use_container_width=True
            )

        else:

            st.info(
                """
                The RBF kernel does not provide a direct
                784-pixel coefficient map.

                The SVM decision scores and probabilities
                above show the classification decision.
                """
            )


    # ========================================================
    # HUMAN FEEDBACK
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '7️⃣ Was the prediction correct?'
        '</div>',
        unsafe_allow_html=True
    )

    feedback = st.radio(
        "Tell the system:",
        [
            "Yes, prediction is correct",
            "No, prediction is incorrect"
        ],
        key="prediction_feedback"
    )


    # ========================================================
    # CORRECT
    # ========================================================

    if feedback == "Yes, prediction is correct":

        st.success(
            f"""
            ✅ The model correctly classified this image
            as **{prediction}**.
            """
        )


    # ========================================================
    # INCORRECT
    # ========================================================

    else:

        st.warning(
            f"""
            The SVM predicted **{prediction}**.
            If this is incorrect, provide the correct label
            below and add this example to the training data.
            """
        )


        correct_label = st.selectbox(
            "Correct digit:",
            list(range(10)),
            key="correct_digit"
        )


        st.image(
            processed_image,
            width=150,
            caption="Example to be added to training data"
        )


        if st.button(
            "➕ Add corrected example and retrain SVM",
            type="primary",
            key="retrain_button"
        ):

            with st.spinner(
                "Adding example and retraining SVM..."
            ):

                # ------------------------------------------------
                # RETRAIN
                # ------------------------------------------------

                (
                    new_model,
                    new_X_train,
                    new_y_train,
                    new_X_custom,
                    new_y_custom
                ) = retrain_with_new_example(
                    features,
                    correct_label,
                    X_train,
                    y_train,
                    X_custom,
                    y_custom,
                    model_path=MODEL_PATH,
                    custom_data_path=CUSTOM_DATA_PATH
                )


                # ------------------------------------------------
                # Save correction information
                # ------------------------------------------------

                st.session_state.correction_message = {
                    "old_prediction": prediction,
                    "correct_label": correct_label
                }


                # ------------------------------------------------
                # Invalidate cached model
                # ------------------------------------------------

                st.session_state.model_version += 1


            st.success(
                f"""
                ✅ Example labelled as **{correct_label}**
                and added to the training data.

                The SVM has been retrained.
                """
            )


            st.rerun()


    # ========================================================
    # CORRECTION RESULT
    # ========================================================

    if st.session_state.correction_message is not None:

        correction = \
            st.session_state.correction_message

        old_prediction = \
            correction["old_prediction"]

        correct_label = \
            correction["correct_label"]


        st.markdown(
            '<div class="section-title">'
            '8️⃣ After retraining'
            '</div>',
            unsafe_allow_html=True
        )


        # ----------------------------------------------------
        # Load latest model
        # ----------------------------------------------------

        (
            latest_model,
            latest_X_train,
            latest_y_train,
            latest_X_test,
            latest_y_test,
            latest_X_custom,
            latest_y_custom
        ) = load_model(
            st.session_state.model_version
        )


        # ----------------------------------------------------
        # Predict again
        # ----------------------------------------------------

        new_prediction, new_probabilities = \
            predict_digit(
                latest_model,
                features
            )


        # ----------------------------------------------------
        # Display comparison
        # ----------------------------------------------------

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Previous prediction",
                old_prediction
            )

        with col2:

            st.metric(
                "Correct label",
                correct_label
            )

        with col3:

            st.metric(
                "New prediction",
                new_prediction
            )


        if new_prediction == correct_label:

            st.success(
                f"""
                🎉 The retrained SVM now predicts
                **{new_prediction}**, which matches the
                corrected label.
                """
            )

        else:

            st.warning(
                f"""
                The retrained SVM predicts
                **{new_prediction}**, while the correct
                label is **{correct_label}**.

                This demonstrates that adding a single
                example does not necessarily change the
                decision boundary enough to change a
                prediction.
                """
            )


        # ----------------------------------------------------
        # Updated probabilities
        # ----------------------------------------------------

        st.subheader(
            "Updated SVM probabilities"
        )

        fig = plot_probabilities(
            new_probabilities,
            new_prediction
        )

        st.pyplot(
            fig,
            use_container_width=True
        )


        # ----------------------------------------------------
        # Clear correction
        # ----------------------------------------------------

        if st.button(
            "Clear correction message"
        ):

            st.session_state.correction_message = None

            st.rerun()


# ============================================================
# INITIAL SCREEN
# ============================================================

else:

    st.info(
        "👆 Upload or capture a handwritten digit to begin."
    )

    st.markdown(
        """
        ### 🎓 Demonstration pipeline

        ```text
        Handwritten digit
                ↓
        Image preprocessing
                ↓
        28 × 28 pixels
                ↓
        784 features
                ↓
        SVM
                ↓
        Decision boundary
                ↓
        Class probabilities
                ↓
        Prediction
                ↓
        Human feedback
                ↓
        Correct label
                ↓
        Retrain SVM
        ```
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "MNIST SVM Demonstration | "
    "Support Vector Machine + Human-in-the-Loop Learning"
)
