import os
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

from utils.preprocessing import preprocess_image
from utils.visualization import (
    plot_prediction_probabilities,
    plot_feature_maps
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="MNIST Digit Classifier",
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
        font-size: 42px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #666;
        margin-bottom: 30px;
    }

    .prediction-box {
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        background-color: #f1f5f9;
        margin-top: 20px;
    }

    .prediction-digit {
        font-size: 80px;
        font-weight: 800;
    }

    .section-title {
        font-size: 26px;
        font-weight: 700;
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
    '<div class="main-title">🔢 MNIST Handwritten Digit Classifier</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Upload or capture a handwritten digit and see how a CNN classifies it.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# MODEL
# ============================================================

MODEL_PATH = "mnist_cnn.keras"


@st.cache_resource
def load_model():

    if os.path.exists(MODEL_PATH):

        model = tf.keras.models.load_model(MODEL_PATH)

    else:

        st.info(
            "The trained model was not found. "
            "Training a CNN on MNIST for the first time..."
        )

        from train_model import train_model

        model = train_model(
            epochs=3,
            save_path=MODEL_PATH
        )

    return model


with st.spinner("Loading MNIST classifier..."):
    model = load_model()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Demonstration Controls")

st.sidebar.write(
    """
    This application demonstrates the complete machine-learning
    pipeline:

    **Input → Preprocessing → CNN → Probabilities → Prediction**
    """
)

show_processing = st.sidebar.checkbox(
    "Show preprocessing steps",
    value=True
)

show_feature_maps = st.sidebar.checkbox(
    "Show CNN feature maps",
    value=True
)

show_probabilities = st.sidebar.checkbox(
    "Show class probabilities",
    value=True
)


# ============================================================
# INPUT METHOD
# ============================================================

st.markdown(
    '<div class="section-title">1️⃣ Provide a handwritten digit</div>',
    unsafe_allow_html=True
)

input_method = st.radio(
    "Choose input method:",
    [
        "Upload an image",
        "Use camera"
    ],
    horizontal=True
)

image = None


# ============================================================
# UPLOAD
# ============================================================

if input_method == "Upload an image":

    uploaded_file = st.file_uploader(
        "Upload a handwritten digit image",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("L")


# ============================================================
# CAMERA
# ============================================================

else:

    camera_image = st.camera_input(
        "Take a picture of a handwritten digit"
    )

    if camera_image is not None:

        image = Image.open(camera_image).convert("L")


# ============================================================
# CLASSIFICATION
# ============================================================

if image is not None:

    st.markdown(
        '<div class="section-title">2️⃣ Input image</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    # --------------------------------------------------------
    # ORIGINAL IMAGE
    # --------------------------------------------------------

    with col1:

        st.subheader("Original image")

        st.image(
            image,
            width=300
        )

    # --------------------------------------------------------
    # PREPROCESSING
    # --------------------------------------------------------

    processed_image, processing_steps = preprocess_image(
        image,
        return_steps=True
    )

    with col2:

        st.subheader("Processed MNIST image")

        st.image(
            processed_image,
            width=300,
            clamp=True
        )


    # ========================================================
    # SHOW PREPROCESSING
    # ========================================================

    if show_processing:

        st.markdown(
            '<div class="section-title">'
            '3️⃣ How the image is prepared'
            '</div>',
            unsafe_allow_html=True
        )

        st.write(
            """
            A real-world photograph is not directly suitable for the
            MNIST model. The image is converted into the format expected
            by the neural network.
            """
        )

        pcols = st.columns(4)

        step_names = [
            "Grayscale",
            "Invert / Normalize",
            "Crop digit",
            "Resize to 28 × 28"
        ]

        for i, (name, step_image) in enumerate(
            zip(step_names, processing_steps)
        ):

            with pcols[i]:

                st.caption(name)

                st.image(
                    step_image,
                    use_container_width=True
                )


    # ========================================================
    # MODEL INPUT
    # ========================================================

    input_array = np.array(processed_image).astype(
        "float32"
    ) / 255.0

    input_array = input_array.reshape(
        1, 28, 28, 1
    )


    # ========================================================
    # PREDICTION
    # ========================================================

    probabilities = model.predict(
        input_array,
        verbose=0
    )[0]

    prediction = int(
        np.argmax(probabilities)
    )

    confidence = float(
        probabilities[prediction]
    )


    # ========================================================
    # RESULT
    # ========================================================

    st.markdown(
        '<div class="section-title">4️⃣ CNN classification result</div>',
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
        Confidence: <b>{confidence * 100:.2f}%</b>
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
            '<div class="section-title">'
            '5️⃣ What did the CNN think?'
            '</div>',
            unsafe_allow_html=True
        )

        st.write(
            """
            The final CNN layer produces a probability for every
            possible digit from 0 to 9.
            """
        )

        fig = plot_prediction_probabilities(
            probabilities,
            prediction
        )

        st.pyplot(
            fig,
            use_container_width=True
        )

        # ----------------------------------------------------
        # NUMERICAL PROBABILITIES
        # ----------------------------------------------------

        st.subheader("Class probabilities")

        probability_columns = st.columns(5)

        for digit in range(10):

            with probability_columns[digit % 5]:

                st.metric(
                    label=f"Digit {digit}",
                    value=f"{probabilities[digit] * 100:.2f}%"
                )


    # ========================================================
    # CNN FEATURE MAPS
    # ========================================================

    if show_feature_maps:

        st.markdown(
            '<div class="section-title">'
            '6️⃣ What is happening inside the CNN?'
            '</div>',
            unsafe_allow_html=True
        )

        st.write(
            """
            The first convolutional layer learns simple visual
            patterns such as edges, curves and stroke directions.

            The next convolutional layer combines these patterns
            into more meaningful shapes.
            """
        )

        # Find convolutional layers
        conv_layers = [
            layer
            for layer in model.layers
            if isinstance(
                layer,
                tf.keras.layers.Conv2D
            )
        ]

        if len(conv_layers) > 0:

            feature_model = tf.keras.Model(
                inputs=model.input,
                outputs=[
                    layer.output
                    for layer in conv_layers
                ]
            )

            feature_maps = feature_model.predict(
                input_array,
                verbose=0
            )

            layer_names = [
                layer.name
                for layer in conv_layers
            ]

            selected_layer = st.selectbox(
                "Select convolution layer:",
                range(len(layer_names)),
                format_func=lambda x: layer_names[x]
            )

            selected_maps = feature_maps[selected_layer]

            fig = plot_feature_maps(
                selected_maps,
                max_maps=16
            )

            st.pyplot(
                fig,
                use_container_width=True
            )

        else:

            st.warning(
                "No convolutional layers were found."
            )


    # ========================================================
    # SIMPLE EXPLANATION
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '🧠 How did the classifier make the decision?'
        '</div>',
        unsafe_allow_html=True
    )

    st.info(
        f"""
        **Step 1:** The input image was converted to grayscale.

        **Step 2:** The digit was normalized and resized to 28 × 28 pixels.

        **Step 3:** The CNN convolution layers detected visual patterns
        such as edges, curves and strokes.

        **Step 4:** Deeper layers combined these patterns into
        digit-specific features.

        **Step 5:** The final Softmax layer produced probabilities
        for digits 0–9.

        **Step 6:** The digit with the highest probability was selected.

        Therefore, the CNN predicted **{prediction}**
        with a probability of **{confidence * 100:.2f}%**.
        """
    )


# ============================================================
# INITIAL SCREEN
# ============================================================

else:

    st.info(
        "👆 Upload an image or use the camera to begin."
    )

    st.markdown(
        """
        ### Recommended input

        For the best demonstration:

        - Write **one digit** using a thick black pen.
        - Use a **white sheet of paper**.
        - Keep the digit approximately centered.
        - Avoid shadows where possible.
        - Take the photograph from directly above.

        The application will automatically convert the photograph
        into the 28 × 28 format used by MNIST.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "MNIST CNN Demonstration • Built with Streamlit and TensorFlow"
)
