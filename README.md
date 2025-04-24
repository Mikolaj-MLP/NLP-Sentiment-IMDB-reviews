*Overview*
This repo demonstrates a complete workflow for training, evaluating, and comparing deep learning models on a binary sentiment classification task. The primary objectives are:

Implement four neural architectures (Unidirectional LSTM, Bidirectional LSTM, Attention-Based Model, TextCNN) for classifying movie reviews as positive or negative.

Evaluate each model using training, validation (with early stopping), and test splits.

Compare performance metrics (loss curves, accuracy, F1, confusion matrices) across models.

Explore post‑training calibration and thresholding to analyze trade‑offs between confidence and coverage.

Data

Splits: 40,000 reviews for training, 5,000 for validation, 5,000 for testing.
