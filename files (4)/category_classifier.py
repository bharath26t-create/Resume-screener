"""
category_classifier.py

Loads the locally-trained TF-IDF + LinearSVC model (see
train_category_classifier.py) and uses it to predict a resume's job
category. This runs entirely offline, with no API call and no cost —
separate from the GPT-based extraction/scoring in matcher.py.
"""

import os
import joblib

MODEL_PATH = "category_classifier.pkl"

_model = None


def load_classifier():
    global _model
    if _model is not None:
        return _model

    if not os.path.isfile(MODEL_PATH):
        print(f"[category_classifier] '{MODEL_PATH}' not found. "
              f"Run train_category_classifier.py first.")
        return None

    _model = joblib.load(MODEL_PATH)
    return _model


def predict_category(resume_text):
    """
    Returns (predicted_category, confidence) or (None, None) if no
    model is available. Confidence here is the decision-function margin
    for the winning class, normalized to a rough 0-1 scale — LinearSVC
    doesn't give true probabilities without extra calibration, so treat
    this as a relative confidence signal, not a calibrated percentage.
    """
    model = load_classifier()
    if model is None:
        return None, None

    prediction = model.predict([resume_text])[0]

    try:
        scores = model.decision_function([resume_text])[0]
        margin = float(max(scores) - sorted(scores)[-2])
        confidence = round(min(margin / 2.0, 1.0), 3)
    except Exception:
        confidence = None

    return prediction, confidence
