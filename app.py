import os
import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt

from utils.preprocessing import preprocess_image, make_debug_figure
from utils.model_utils import load_model, predict_with_activations


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="MNIST Digit Classifier",
    page_icon="🔢",
    layout="wide",
)


MODEL_PATH = "models/mnist_cnn.keras"


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def get_model():
    return load_model(MODEL_PATH)


# ============================================================
# PROBABILITY GRAPH
# ============================================================

def plot_probabilities(probabilities):

    fig, ax = plt.subplots(figsize=(8, 4))

    digits = list(range(10))

    ax.bar(digits, probabilities)

    ax.set_xticks(digits)
    ax.set_xlabel("Digit")
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1)

    ax.set_title("CNN Softmax Output")

    fig.tight_layout()

    return fig


# ============================================================
# EXPLANATION
# ============================================================

def explain_prediction(digit, confidence):

    return (
        f"The CNN predicts **{digit}** because the learned convolution "
        f"filters detected stroke patterns that are most similar to examples "
        f"of digit **{digit}** in the MNIST dataset. The final Softmax layer "
        f"assigns the highest probability to class **{digit}**, with a "
        f"confidence of approximately **{confidence:.2%}**."
    )


# ============================================================
# HEADER
# ============================================================

st.title("🔢 MNIST Handwritten Digit Classifier")

st.markdown(
    """
### An interactive demonstration of how a CNN recognizes handwritten digits

Write a digit, upload its image or capture it using your camera,
and observe how the CNN processes and classifies it.
"""
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🎓 Student Demonstration")

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
