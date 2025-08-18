import argparse
from pathlib import Path

import joblib
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_curve, classification_report, average_precision_score, roc_auc_score, \
    f1_score


def evaluate_models(model_paths, X_test_path, y_test_path):
    fig, ax = plt.subplots(figsize=(10, 7))

    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path)

    print(f"Loaded test data. X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    results = []

    for path in model_paths:
        model_path = Path(path)

        if not model_path.exists():
            print(f"Model file not found at {model_path}. Skipping...")
            continue

        print(f"Evaluating Model: {model_path.name}")
        model = joblib.load(model_path)

        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        report = classification_report(y_test, y_pred, target_names=['Non Fraud', 'Fraud'])
        print('Classification Report:')
        print(report)

        auprc = average_precision_score(y_test, y_pred_proba)
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        results.append({
            'Model': model_path.stem,
            'AURPC': auprc,
            'ROC AUC': roc_auc_score(y_test, y_pred_proba),
            'Recall (Fraud)': recall,
            'Precision (Fraud)': precision,
            'F1-Score (Fraud)': f1_score(y_test, y_pred, pos_label=1)
        })

        ax.plot(recall, precision, label=f'{model_path.stem} (AUPRC = {auprc:.3f})')

    if results:
        print(f'Model Comparison Summary')
        results_df = pd.DataFrame(results).set_index('Model')
        print(results_df.round(4))


    ax.set_xlabel('Recall', fontsize=14)
    ax.set_ylabel('Precision', fontsize=14)
    ax.set_title('Precision-Recall Curve for Fraud Detection', fontsize=16)
    ax.legend(loc='lower left')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate and compare trained models')
    parser.add_argument('--models', nargs="+", required=True, help='Paths to model files')
    parser.add_argument('--x-test', required=True, help='Paths to X_test_scaled.csv')
    parser.add_argument('--y-test', required=True, help='Paths to y_test.csv')

    args = parser.parse_args()
    evaluate_models(args.models, args.x_test, args.y_test)



