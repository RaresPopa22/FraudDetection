from src.data_processing import feature_engineer
from src.data_processing import preprocess_data
from sklearn.model_selection import train_test_split


class TestFeatureEngineer:

    @staticmethod
    def check_multiple_features(expected: list[str], result):
        for feature in expected:
            assert feature in result.columns, f'Missing {feature}'

    def test_removes_column(self, sample_raw_data):
        result = feature_engineer(sample_raw_data.copy())
        assert 'Time' not in result.columns

    def test_creates_time_cyclical_features(self, sample_raw_data):
        result = feature_engineer(sample_raw_data.copy())
        assert 'Hour_sin' in result.columns
        assert 'Hour_cos' in result.columns

    def test_interaction_features(self, sample_raw_data):
        result = feature_engineer(sample_raw_data.copy())
        expected = ['V17_V14', 'V17_V14_hour_sin', 'V17_V14_hour_cos', 'V12_V10', 'top_features']

        self.check_multiple_features(expected, result)

    def test_polynomial_features(self, sample_raw_data):
        result = feature_engineer(sample_raw_data.copy())
        expected = ['V17_squared', 'V14_squared', 'V12_squared', 'top_features_squared']

        self.check_multiple_features(expected, result)

    def test_aggregate_statistics(self, sample_raw_data):
        result = feature_engineer(sample_raw_data.copy())
        expected = ['top_mean', 'top_min', 'top_max', 'top_median', 'top_sum']

        self.check_multiple_features(expected, result)


class TestPreprocessData:

    def test_removes_column(self, tmp_path, sample_raw_data):
        data_path = tmp_path / "test_data.csv"
        sample_raw_data.to_csv(data_path, index=False)

        config = {
            'data_paths': {'raw_data': str(data_path)},
            'preprocessing': {'eval_size': 0.15}
        }

        X_train, X_test, y_train, y_test, scaler = preprocess_data(config)
        assert 'Amount' not in X_train.columns
        assert 'Amount' not in X_test.columns

    def test_removes_target_from_X_data(self, tmp_path, sample_raw_data):
        data_path = tmp_path / "test_data.csv"
        sample_raw_data.to_csv(data_path, index=False)

        config = {
            'data_paths': {'raw_data': str(data_path)},
            'preprocessing': {'eval_size': 0.15}
        }

        X_train, X_test, y_train, y_test, scaler = preprocess_data(config)

        assert 'Class' not in X_train.columns
        assert 'Class' not in X_test.columns


    def test_scaler_fitted_on_train_dataset_only(self, tmp_path, sample_raw_data):
        data_path = tmp_path / "test_data.csv"
        sample_raw_data.to_csv(data_path, index=False)

        config = {
            'data_paths': {'raw_data': str(data_path)},
            'preprocessing': {'eval_size': 0.15}
        }

        X_train, X_test, y_train, y_test, scaler = preprocess_data(config)

        data = sample_raw_data.copy()
        data = feature_engineer(data)
        y = data['Class']
        X = data.drop('Class', axis=1)
        X_train_expected, X_test_expected, _, _ = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

        expected_mean = X_train_expected['Amount'].mean()
        expected_std = X_train_expected['Amount'].std(ddof=0)

        assert abs(scaler.mean_[0] - expected_mean) < 0.01
        assert abs(scaler.scale_[0] - expected_std) < 0.01

        assert 'scaled_amount' in X_train.columns
        assert 'scaled_amount' in X_test.columns
