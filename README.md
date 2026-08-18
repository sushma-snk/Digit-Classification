# 🔢 MNIST Handwritten Digit Classifier

An interactive Streamlit application for demonstrating handwritten digit classification using a Convolutional Neural Network (CNN).

Students can:

- Upload a handwritten digit image
- Capture a digit using the camera
- See the preprocessing pipeline
- See the final prediction
- See probabilities for all ten digits
- Visualize CNN feature maps
- Understand how convolutional layers extract features

---

## Machine Learning Pipeline

The application demonstrates:

Image

↓

Grayscale conversion

↓

Normalization

↓

Digit extraction

↓

28 × 28 resizing

↓

CNN convolution layers

↓

Feature extraction

↓

Dense layer

↓

Softmax probabilities

↓

Predicted digit

---

## CNN Architecture

The classifier uses:

- Conv2D – 32 filters
- MaxPooling2D
- Conv2D – 64 filters
- MaxPooling2D
- Flatten
- Dense – 128 neurons
- Dropout
- Dense – 10 neurons with Softmax

---

## Project Structure

```text
mnist-digit-classifier/
│
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── .gitignore
│
└── utils/
    ├── __init__.py
    ├── preprocessing.py
    └── visualization.py
