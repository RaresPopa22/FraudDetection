import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import (precision_recall_curve, classification_report, average_precision_score,
                             roc_auc_score, f1_score, recall_score, precision_score)

from src.data_processing import load_test_data, read_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def find_optimal_threshold(y_true, y_pred_proba):
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-8)
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx]


def plot_precision_recall_curve(recalls, precisions, labels, auprcs):
    fig, ax = plt.subplots(figsize=(10, 7))

    for i in range(len(recalls)):
        ax.plot(recalls[i], precisions[i], label=f'{labels[i]} (AUPRC = {auprcs[i]:.3f})')

    ax.set_xlabel('Recall', fontsize=14)
    ax.set_ylabel('Precision', fontsize=14)
    ax.set_title('Precision-Recall Curve for Fraud Detection', fontsize=16)
    ax.legend(loc='lower left')
    plt.show()


def evaluate_models(base_config, model_paths):
    X_test, y_test = load_test_data(base_config)

    logger.info(f"Loaded test data. X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    results = []
    precisions = []
    recalls = []
    labels = []
    auprcs = []

    res = {}

    for path in model_paths:
        model_path = Path(path)

        base_config['model_name'] = model_path.stem

        if not model_path.exists():
            logger.info(f"Model file not found at {model_path}. Skipping...")
            continue

        logger.info(f"Evaluating Model: {model_path.name}")

        model = joblib.load(model_path)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        res[model_path.stem] = y_pred_proba

        threshold = find_optimal_threshold(y_test, y_pred_proba)
        threshold_path =  'models/' + model_path.stem + '_threshold.json'
        with open(threshold_path, 'w') as f:
            json.dump({'threshold': float(threshold)}, f)

        y_pred = (y_pred_proba >= threshold).astype(int)
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        auprc = average_precision_score(y_test, y_pred_proba)
        auprcs.append(auprc)

        report = classification_report(y_test, y_pred, target_names=['Non Fraud', 'Fraud'])
        logger.info('Classification Report:')
        logger.info(report)

        recalls.append(recall)
        precisions.append(precision)
        labels.append(model_path.stem)

        results.append({
            'Model': model_path.stem,
            'AUPRC': auprc,
            'ROC AUC': roc_auc_score(y_test, y_pred_proba),
            'Recall (Fraud)': recall_score(y_test, y_pred, pos_label=1),
            'Precision (Fraud)': precision_score(y_test, y_pred, pos_label=1),
            'F1-Score (Fraud)': f1_score(y_test, y_pred, pos_label=1)
        })

    if results:
        logger.info(f'Model Comparison Summary')
        results_df = pd.DataFrame(results).set_index('Model')
        logger.info(results_df.round(4))

    plot_precision_recall_curve(recalls, precisions, labels, auprcs)


if __name__ == '__main__':
    base_config_path = Path(__file__).parent.parent / 'config' / 'base_config.yaml'
    base_config = read_config(base_config_path)
    parser = argparse.ArgumentParser(description='Evaluate and compare trained models')
    parser.add_argument('--models', nargs="+", required=True, help='Paths to model files')

    args = parser.parse_args()
    evaluate_models(base_config, args.models)
