import argparse
import logging

import joblib
import pandas as pd

from src.data_processing import feature_engineer

logger = logging.getLogger(__name__)


def predict(model_path, scalar_path, input_path, threshold=0.5):
    model = joblib.load(model_path)
    scaler = joblib.load(scalar_path)

    data = pd.read_csv(input_path)
    data['scaled_amount'] = scaler.transform(data['Amount'].values.reshape(-1, 1))
    data = feature_engineer(data)
    data = data.drop('Amount', axis=1)

    if 'Class' in data.columns:
        data = data.drop('Class', axis=1)

    y_pred_proba = model.predict_proba(data)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)

    return y_pred, y_pred_proba


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run fraud detection on new data")
    parser.add_argument('--input', required=True, help='Path to input CSV')
    parser.add_argument('--model', required=True, help='Path to trained model')
    parser.add_argument('--scaler', required=True, help='Path to fitted scalar')
    parser.add_argument('--threshold', type=float, default=0.5, help='Classification threshold')
    parser.add_argument('--output', required=True, help='Path to save prediction in CSV format')
    args = parser.parse_args()

    preds, probas = predict(args.model, args.scalar, args.input, args.threshold)