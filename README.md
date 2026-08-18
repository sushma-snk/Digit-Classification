# 🔢 MNIST Handwritten Digit Classifier

An interactive Streamlit application for demonstrating handwritten digit classification using the K-Nearest Neighbors (KNN) machine learning algorithm.

Students can:

- Upload a handwritten digit
- Capture a handwritten digit using a camera
- Observe image preprocessing
- See the 784 pixel features
- Change the value of K
- See the nearest MNIST training examples
- Observe the voting process
- See the final classification

---

## Machine Learning Pipeline

```text
Handwritten Image
       ↓
Grayscale Conversion
       ↓
Normalization
       ↓
Digit Extraction
       ↓
Resize to 28 × 28
       ↓
Flatten
       ↓
784 Features
       ↓
KNN
       ↓
Distance Calculation
       ↓
K Nearest Neighbors
       ↓
Majority Voting
       ↓
Predicted Digit
