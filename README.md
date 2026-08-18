# 🔢 MNIST Handwritten Digit Classifier

An interactive Streamlit application for demonstrating handwritten digit classification using a Support Vector Machine (SVM).

The application allows students to:

- Upload handwritten digit images
- Capture handwritten digits using a camera
- Observe image preprocessing
- See the 28 × 28 MNIST representation
- View SVM class probabilities
- Provide feedback when the model is incorrect
- Add corrected examples to the training dataset
- Retrain the SVM
- Compare predictions before and after retraining

---

## Machine Learning Pipeline

```text
Handwritten Image
       ↓
Grayscale
       ↓
Normalization
       ↓
Digit extraction
       ↓
28 × 28 image
       ↓
784 pixel features
       ↓
SVM
       ↓
Decision function
       ↓
Class probabilities
       ↓
Predicted digit
