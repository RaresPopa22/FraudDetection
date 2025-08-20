import argparse
import joblib
import pandas as pd
import tensorflow as tf

from pathlib import Path
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_curve, classification_report, average_precision_score, roc_auc_score, \
    f1_score, auc
from tensorflow.keras.metrics import Precision, Recall

from src.data_processing import load_test_data, read_config


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

    print(f"Loaded test data. X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    results = []
    precisions = []
    recalls = []
    labels = []
    auprcs = []

    for path in model_paths:
        model_path = Path(path)
        base_config['model_name'] = model_path.stem

        if not model_path.exists():
            print(f"Model file not found at {model_path}. Skipping...")
            continue

        print(f"Evaluating Model: {model_path.name}")

        model = joblib.load(model_path)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        auprc = average_precision_score(y_test, y_pred_proba)
        auprcs.append(auprc)

        report = classification_report(y_test, y_pred, target_names=['Non Fraud', 'Fraud'])
        print('Classification Report:')
        print(report)

        recalls.append(recall)
        precisions.append(precision)
        labels.append(model_path.stem)

        results.append({
            'Model': model_path.stem,
            'AURPC': auprc,
            'ROC AUC': roc_auc_score(y_test, y_pred_proba),
            'Recall (Fraud)': recall,
            'Precision (Fraud)': precision,
            'F1-Score (Fraud)': f1_score(y_test, y_pred, pos_label=1)
        })

    if results:
        print(f'Model Comparison Summary')
        results_df = pd.DataFrame(results).set_index('Model')
        print(results_df.round(4))

    plot_precision_recall_curve(recalls, precisions, labels, auprcs)

if __name__ == '__main__':
    base_config = read_config("../config/base_config.yaml")
    parser = argparse.ArgumentParser(description='Evaluate and compare trained models')
    parser.add_argument('--models', nargs="+", required=True, help='Paths to model files')
    parser.add_argument('--x-test', required=True, help='Paths to X_test_scaled.csv')
    parser.add_argument('--y-test', required=True, help='Paths to y_test.csv')

    args = parser.parse_args()
    evaluate_models(base_config, args.models)



