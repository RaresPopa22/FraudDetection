from pathlib import Path

import joblib
import lightgbm as lgb
from pandas import DataFrame

from src.data_processing import feature_engineer
from sklearn.preprocessing import StandardScaler

from src.predict import predict


class TestPredict:

    @staticmethod
    def setup(sample_raw_data: DataFrame, tmp_path: Path):
        data = sample_raw_data.copy()
        y = data['Class']
        data = feature_engineer(data)
        scaler = StandardScaler()
        data['scaled_amount'] = scaler.fit_transform(data[['Amount']])
        data = data.drop(['Amount', 'Class'], axis=1)

        model = lgb.LGBMClassifier(n_estimators=10, random_state=1)
        model.fit(data, y)

        model_path = tmp_path / "model.joblib"
        scaler_path = tmp_path / "scaler.joblib"
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)

        input_path = tmp_path / "input.csv"
        test_input = sample_raw_data.drop('Class', axis=1).head(10)
        test_input.to_csv(input_path, index=False)
        return str(model_path), str(scaler_path), str(input_path)

    def test_predict_returns_binary_and_proba(self, tmp_path, sample_raw_data):
        input_path, model_path, scaler_path = self.setup(sample_raw_data, tmp_path)

        preds, probas = predict(input_path, model_path, scaler_path)

        assert len(preds) == 10
        assert len(probas) == 10
        assert set(preds).issubset({0, 1})
        assert all(0 <= p <= 1 for p in probas)

    def test_threshold_works(self, tmp_path, sample_raw_data):
        input_path, model_path, scaler_path = self.setup(sample_raw_data, tmp_path)

        preds_low, _ = predict(input_path, model_path, scaler_path, 0.1)
        preds_high, _ = predict(input_path, model_path, scaler_path, 0.9)

        assert sum(preds_low) >= sum(preds_high)
