# Human vs AI — MNIST Digit Challenge

Interactive Streamlit classroom demo: student writes a digit, CNN predicts it, student verifies it, and wrong examples can be used for a small in-memory fine-tuning step.

## Run locally
Use Python 3.12 (recommended for Streamlit Cloud).

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

The first `train_model.py` run downloads MNIST once and creates `models/mnist_cnn.keras`. Commit that model file to GitHub if deploying to Streamlit Cloud, so Cloud does not need to train during deployment.

## Classroom flow
1. Write one digit on paper.
2. Upload or capture it.
3. Ask AI to classify.
4. Click right/wrong.
5. If wrong, enter the actual label.
6. Fine-tune the running model.
7. Try another handwritten digit.

## Important
This is a teaching demonstration. The correction is used for a tiny fine-tuning step on the in-memory model; it is not full retraining from scratch and the original model file is not overwritten.
