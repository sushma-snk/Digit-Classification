import os
import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt

from utils.preprocessing import (
    preprocess_image,
    make_debug_figure,
)

from utils.model_utils import (
    load_model,
    predict_with_activations,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="MNIST Digit Classifier",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CONSTANTS
# ============================================================

MODEL_PATH = "models/mnist_cnn.keras"


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
        background-color: #f5f7ff;
        text-align: center;
        border: 1px solid #dfe4ff;
    }

    .prediction-digit {
        font-size: 90px;
        font-weight: 800;
    }

    .prediction-label {
        font-size: 18px;
        color: #555;
    }

    .info-card {
        padding: 20px;
        border-radius: 12px;
        background-color: #f7f7f7;
        margin-bottom: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def get_model():

    return load_model(
        MODEL_PATH
    )


# ============================================================
# PROBABILITY GRAPH
# ============================================================

def plot_probabilities(
    probabilities,
    predicted_digit,
):

    fig, ax = plt.subplots(
        figsize=(9, 4.5)
    )

    digits = np.arange(10)

    bars = ax.bar(
        digits,
        probabilities,
    )

    # Highlight predicted digit
    bars[predicted_digit].set_alpha(
        1.0
    )

    ax.set_xticks(
        digits
    )

    ax.set_xlabel(
        "Digit",
        fontsize=12,
    )

    ax.set_ylabel(
        "Probability",
        fontsize=12,
    )

    ax.set_ylim(
        0,
        1.05,
    )

    ax.set_title(
        "CNN Softmax Probabilities",
        fontsize=15,
        fontweight="bold",
    )

    # Display values above bars
    for digit, probability in enumerate(
        probabilities
    ):

        if probability > 0.01:

            ax.text(
                digit,
                probability + 0.02,
                f"{probability:.1%}",
                ha="center",
                fontsize=9,
            )

    fig.tight_layout()

    return fig


# ============================================================
# PREDICTION EXPLANATION
# ============================================================

def explain_prediction(
    digit,
    confidence,
):

    if confidence >= 0.90:

        confidence_text = (
            "The model is highly confident."
        )

    elif confidence >= 0.70:

        confidence_text = (
            "The model has reasonably high confidence."
        )

    elif confidence >= 0.50:

        confidence_text = (
            "The model is somewhat uncertain."
        )

    else:

        confidence_text = (
            "The model is uncertain about the prediction."
        )

    return f"""
### 🤖 Why did the CNN predict {digit}?

The CNN processed the 28 × 28 image and extracted
patterns such as **edges, curves and strokes** using
convolution filters.

These learned features were passed through the deeper
layers of the network.

Finally, the **Softmax layer** calculated a probability
for each digit from 0 to 9.

The highest probability corresponds to **digit {digit}**.

**Confidence: {confidence:.2%}**

{confidence_text}
"""


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🔢 MNIST Digit Classifier</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
    An interactive demonstration of handwritten digit
    classification using a Convolutional Neural Network
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "🎓 Student Demo"
    )

    st.markdown(
        """
        ### Classification Pipeline

        ```text
        Handwritten Image
                ↓
        Preprocessing
                ↓
        28 × 28 Image
                ↓
        Convolution
                ↓
        Feature Maps
                ↓
        Pooling
                ↓
        Dense Layer
                ↓
        Softmax
                ↓
        Prediction
        ```
        """
    )

    st.divider()

    st.subheader(
        "🧠 Model Information"
    )

    st.write(
        "**Dataset:** MNIST"
    )

    st.write(
        "**Input:** 28 × 28 grayscale"
    )

    st.write(
        "**Classes:** 10 digits"
    )

    st.write(
        "**Classifier:** CNN"
    )

    st.divider()

    st.caption(
        "MNIST Digit Classification Demo"
    )


# ============================================================
# CHECK MODEL
# ============================================================

if not os.path.exists(
    MODEL_PATH
):

    st.error(
        """
        ### ❌ Trained model not found

        The application expected the trained model at:

        `models/mnist_cnn.keras`

        Please run:

        ```bash
        python train_model.py
        ```

        and make sure the generated model is uploaded to
        your GitHub repository.
        """
    )

    st.stop()


# ============================================================
# LOAD MODEL
# ============================================================

try:

    model = get_model()

except Exception as error:

    st.error(
        "Unable to load the trained CNN model."
    )

    st.exception(
        error
    )

    st.stop()


# ============================================================
# TABS
# ============================================================

tab_classify, tab_cnn, tab_teaching = st.tabs(
    [
        "🖼️ Classify Digit",
        "🧠 How CNN Works",
        "📚 Teaching Mode",
    ]
)


# ============================================================
# TAB 1
# CLASSIFY DIGIT
# ============================================================

with tab_classify:

    st.header(
        "🖼️ Classify a Handwritten Digit"
    )

    st.write(
        """
        Upload an image or use your camera to provide a
        handwritten digit.
        """
    )


    # --------------------------------------------------------
    # INPUT METHOD
    # --------------------------------------------------------

    input_method = st.radio(
        "Choose input method:",
        [
            "📁 Upload Image",
            "📷 Camera",
        ],
        horizontal=True,
    )


    uploaded_file = None


    # --------------------------------------------------------
    # UPLOAD IMAGE
    # --------------------------------------------------------

    if input_method == "📁 Upload Image":

        uploaded_file = st.file_uploader(
            "Upload a handwritten digit",
            type=[
                "png",
                "jpg",
                "jpeg",
            ],
            help=(
                "For best results, use one large handwritten "
                "digit on a plain background."
            ),
        )


    # --------------------------------------------------------
    # CAMERA INPUT
    # --------------------------------------------------------

    else:

        st.info(
            """
            Write one large digit on a piece of white paper,
            hold it in front of the camera, and capture it.
            """
        )

        uploaded_file = st.camera_input(
            "📷 Capture your handwritten digit"
        )


    # ========================================================
    # IMAGE RECEIVED
    # ========================================================

    if uploaded_file is not None:

        try:

            image = Image.open(
                uploaded_file
            ).convert(
                "RGB"
            )

        except Exception:

            st.error(
                "Unable to read the image."
            )

            st.stop()


        st.divider()


        # ====================================================
        # ORIGINAL IMAGE
        # ====================================================

        st.subheader(
            "1️⃣ Original Image"
        )


        col_original, col_processed = st.columns(
            2
        )


        with col_original:

            st.image(
                image,
                caption="Original input",
                use_container_width=True,
            )


        # ====================================================
        # PREPROCESSING
        # ====================================================

        try:

            processed_image, debug_images = (
                preprocess_image(
                    image
                )
            )

        except Exception as error:

            st.error(
                "Error during image preprocessing."
            )

            st.exception(
                error
            )

            st.stop()


        with col_processed:

            st.image(
                processed_image,
                caption="28 × 28 CNN input",
                width=280,
            )


        st.success(
            """
            The image has been converted into a
            **28 × 28 grayscale image**, similar to
            the format used by MNIST.
            """
        )


        # ====================================================
        # CLASSIFICATION
        # ====================================================

        st.divider()

        st.subheader(
            "2️⃣ Classification"
        )


        classify_button = st.button(
            "🚀 CLASSIFY DIGIT",
            type="primary",
            use_container_width=True,
        )


        if classify_button:

            # ----------------------------------------------
            # PIL → NUMPY
            # ----------------------------------------------

            image_array = np.array(
                processed_image,
                dtype=np.float32,
            )


            # ----------------------------------------------
            # NORMALIZATION
            # ----------------------------------------------

            image_array = (
                image_array / 255.0
            )


            # ----------------------------------------------
            # ADD CNN DIMENSIONS
            # ----------------------------------------------

            image_array = image_array.reshape(
                1,
                28,
                28,
                1,
            )


            # ----------------------------------------------
            # PREDICTION
            # ----------------------------------------------

            try:

                probabilities, activations = (
                    predict_with_activations(
                        model,
                        image_array,
                    )
                )

            except Exception as error:

                st.error(
                    "Error while running the CNN."
                )

                st.exception(
                    error
                )

                st.stop()


            # ----------------------------------------------
            # EXTRACT PROBABILITIES
            # ----------------------------------------------

            probabilities = np.asarray(
                probabilities[0]
            )


            predicted_digit = int(
                np.argmax(
                    probabilities
                )
            )


            confidence = float(
                probabilities[
                    predicted_digit
                ]
            )


            # =================================================
            # RESULT
            # =================================================

            st.divider()

            st.subheader(
                "3️⃣ Prediction"
            )


            result_left, result_right = st.columns(
                [
                    1,
                    2,
                ]
            )


            # -------------------------------------------------
            # PREDICTION
            # -------------------------------------------------

            with result_left:

                st.markdown(
                    f"""
                    <div class="prediction-box">

                    <div class="prediction-label">
                    Predicted Digit
                    </div>

                    <div class="prediction-digit">
                    {predicted_digit}
                    </div>

                    <div class="prediction-label">
                    Confidence: {confidence:.2%}
                    </div>

                    </div>
                    """,
                    unsafe_allow_html=True,
                )


            # -------------------------------------------------
            # PROBABILITY GRAPH
            # -------------------------------------------------

            with result_right:

                st.pyplot(
                    plot_probabilities(
                        probabilities,
                        predicted_digit,
                    ),
                    use_container_width=True,
                )


            # =================================================
            # EXPLANATION
            # =================================================

            st.divider()

            st.markdown(
                explain_prediction(
                    predicted_digit,
                    confidence,
                )
            )


            # =================================================
            # PREPROCESSING DETAILS
            # =================================================

            with st.expander(
                "🔍 See exactly how the image was processed"
            ):

                st.write(
                    """
                    The CNN was not given the original photograph
                    directly.

                    The application first performs several
                    preprocessing operations.
                    """
                )


                preprocessing_fig = (
                    make_debug_figure(
                        image,
                        debug_images,
                    )
                )


                st.pyplot(
                    preprocessing_fig,
                    use_container_width=True,
                )


                st.markdown(
                    """
                    ### Processing sequence

                    **Original image**

                    ↓

                    **Grayscale conversion**

                    ↓

                    **Foreground/background processing**

                    ↓

                    **Digit cropping**

                    ↓

                    **Resize**

                    ↓

                    **Centering**

                    ↓

                    **28 × 28 CNN input**
                    """
                )


            # =================================================
            # FEATURE MAPS
            # =================================================

            with st.expander(
                "🧩 See what the CNN detects"
            ):

                st.markdown(
                    """
                    ### Feature Maps

                    Each convolution filter learns to respond to
                    different visual patterns.

                    Early filters may detect simple edges and
                    strokes. Deeper filters combine these into
                    more meaningful patterns.
                    """
                )


                if len(activations) == 0:

                    st.warning(
                        "No convolution activations are available."
                    )

                else:

                    layer_names = list(
                        activations.keys()
                    )


                    selected_layer = st.selectbox(
                        "Choose a convolution layer:",
                        layer_names,
                    )


                    feature_maps = (
                        activations[
                            selected_layer
                        ][0]
                    )


                    number_of_filters = min(
                        feature_maps.shape[-1],
                        16,
                    )


                    feature_fig, axes = plt.subplots(
                        4,
                        4,
                        figsize=(
                            10,
                            10,
                        ),
                    )


                    axes = axes.flatten()


                    for i in range(16):

                        axes[i].axis(
                            "off"
                        )


                        if i < number_of_filters:

                            axes[i].imshow(
                                feature_maps[
                                    :,
                                    :,
                                    i
                                ],
                                cmap="viridis",
                            )


                            axes[i].set_title(
                                f"Filter {i + 1}"
                            )


                    feature_fig.suptitle(
                        (
                            "CNN Feature Maps — "
                            f"{selected_layer}"
                        ),
                        fontsize=16,
                    )


                    feature_fig.tight_layout()


                    st.pyplot(
                        feature_fig,
                        use_container_width=True,
                    )


            # =================================================
            # ALL PROBABILITIES
            # =================================================

            with st.expander(
                "📊 View numerical probabilities"
            ):

                st.write(
                    "Probability assigned to each digit:"
                )


                for digit in range(10):

                    probability = float(
                        probabilities[
                            digit
                        ]
                    )


                    col_digit, col_bar = st.columns(
                        [
                            1,
                            5,
                        ]
                    )


                    with col_digit:

                        st.write(
                            f"**{digit}**"
                        )


                    with col_bar:

                        st.progress(
                            probability
                        )


                        st.caption(
                            f"{probability:.4%}"
                        )


# ============================================================
# TAB 2
# HOW CNN WORKS
# ============================================================

with tab_cnn:

    st.header(
        "🧠 How Does the CNN Work?"
    )


    st.write(
        """
        This section can be used directly during your classroom
        explanation.
        """
    )


    st.divider()


    st.subheader(
        "The complete classification pipeline"
    )


    st.code(
        """
Handwritten Digit
        │
        ▼
Image Preprocessing
        │
        ▼
28 × 28 Grayscale Image
        │
        ▼
Convolution Layer 1
        │
        ▼
ReLU
        │
        ▼
Max Pooling
        │
        ▼
Convolution Layer 2
        │
        ▼
ReLU
        │
        ▼
Max Pooling
        │
        ▼
Flatten
        │
        ▼
Dense Layer
        │
        ▼
Softmax
        │
        ▼
10 Class Probabilities
        │
        ▼
Predicted Digit
        """,
        language="text",
    )


    st.divider()


    # ========================================================
    # ARCHITECTURE
    # ========================================================

    st.subheader(
        "🏗️ CNN Architecture"
    )


    architecture = {

        "Layer": [

            "Input",

            "Conv2D",

            "MaxPooling2D",

            "Conv2D",

            "MaxPooling2D",

            "Flatten",

            "Dense",

            "Dropout",

            "Output",

        ],

        "Configuration": [

            "28 × 28 × 1",

            "32 filters, 3×3",

            "2 × 2",

            "64 filters, 3×3",

            "2 × 2",

            "1D vector",

            "128 neurons",

            "30%",

            "10 neurons",

        ],

        "Purpose": [

            "Receive image",

            "Detect basic patterns",

            "Reduce image dimensions",

            "Detect complex patterns",

            "Reduce dimensions",

            "Convert feature maps to vector",

            "Combine learned features",

            "Reduce overfitting",

            "Classify digits",

        ],
    }


    st.table(
        architecture
    )


    # ========================================================
    # CONVOLUTION
    # ========================================================

    st.divider()

    st.subheader(
        "🔎 What does convolution do?"
    )


    st.write(
        """
        A convolution filter is a small matrix that moves across
        the image.

        During training, the CNN learns useful filters automatically.

        These filters can learn to detect:

        • Horizontal edges

        • Vertical edges

        • Diagonal strokes

        • Curves

        • Corners

        • Intersections
        """
    )


    st.latex(
        r"""
        Feature\ Map = Input * Filter + Bias
        """
    )


    # ========================================================
    # POOLING
    # ========================================================

    st.divider()

    st.subheader(
        "📉 Why is pooling used?"
    )


    st.write(
        """
        Max pooling reduces the spatial size of feature maps.

        This helps:

        • Reduce computation

        • Reduce the number of parameters

        • Preserve strong features

        • Make the representation less sensitive to small shifts
        """
    )


    # ========================================================
    # SOFTMAX
    # ========================================================

    st.divider()

    st.subheader(
        "🎯 Softmax Classification"
    )


    st.write(
        """
        The final layer contains 10 neurons.

        Each neuron represents one digit:

        **0  1  2  3  4  5  6  7  8  9**
        """
    )


    st.latex(
        r"""
        P(y=i|x)
        =
        \frac{e^{z_i}}
        {\sum_{j=0}^{9}e^{z_j}}
        """
    )


    st.write(
        """
        Softmax converts the final neural-network scores into
        probabilities.

        The digit with the largest probability becomes the
        predicted class.
        """
    )


# ============================================================
# TAB 3
# TEACHING MODE
# ============================================================

with tab_teaching:

    st.header(
        "📚 Teaching Mode"
    )


    st.info(
        """
        ### Important idea

        A CNN does not simply look at the image and "know" the answer.

        It learns useful mathematical representations from many
        training examples.
        """
    )


    st.divider()


    # ========================================================
    # THREE STEPS
    # ========================================================

    col1, col2, col3 = st.columns(
        3
    )


    with col1:

        st.markdown(
            "### 1️⃣ Pixels"
        )

        st.write(
            """
            The handwritten digit is represented as a grid of
            pixel intensity values.
            """
        )


    with col2:

        st.markdown(
            "### 2️⃣ Features"
        )

        st.write(
            """
            Convolution layers learn patterns such as edges,
            curves, corners and strokes.
            """
        )


    with col3:

        st.markdown(
            "### 3️⃣ Decision"
        )

        st.write(
            """
            The network combines these features and produces
            probabilities for the ten possible digits.
            """
        )


    st.divider()


    # ========================================================
    # QUESTIONS
    # ========================================================

    st.subheader(
        "💡 Questions to ask students"
    )


    questions = [

        "Why do we convert the image to grayscale?",

        "Why is the input size 28 × 28?",

        "What is a convolution filter?",

        "What type of patterns can convolution filters detect?",

        "Why do we use ReLU?",

        "Why do we use pooling?",

        "Why are there 10 neurons in the output layer?",

        "What does a Softmax probability represent?",

        "Can a classifier be confident but still wrong?",

        "Why might a camera image be harder to classify than an MNIST image?",

        "What happens if we write two digits in one image?",

    ]


    for number, question in enumerate(
        questions,
        start=1,
    ):

        st.markdown(
            f"**{number}.** {question}"
        )


    st.divider()


    # ========================================================
    # CLASSROOM EXPERIMENT
    # ========================================================

    st.subheader(
        "🧪 Classroom Experiment"
    )


    st.write(
        """
        Ask students to write the same digit in several different
        styles.
        """
    )


    st.markdown(
        """
        For example, ask them to create:

        **7**

        • Normal 7

        • Very large 7

        • Very small 7

        • Tilted 7

        • 7 with a horizontal bar

        • Poorly written 7

        • Thick 7

        • Thin 7
        """
    )


    st.write(
        """
        Upload each version and compare the probability distribution.

        This demonstrates an important machine-learning concept:

        **A model learns from its training distribution, so changes
        in the input can affect its predictions.**
        """
    )


    st.divider()


    st.subheader(
        "🎯 Key takeaway"
    )


    st.success(
        """
        **Machine Learning Pipeline**

        Data → Preprocessing → Feature Learning → Classification → Prediction

        The CNN automatically learns useful features from training
        examples instead of requiring us to manually program rules
        such as "a 7 has this shape".
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🔢 MNIST Digit Classifier | CNN Demonstration | Built with Streamlit"
)
