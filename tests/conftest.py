import pandas as pd
import pytest
import numpy as np


@pytest.fixture
def sample_raw_data():
    np.random.seed(1)
    n_samples = 100

    data = {
        'Time': np.random.randint(0, 3767, n_samples),
        'Amount': np.random.uniform(0, 7700, n_samples),
        'Class': [0] * 98 + [1] * 2
    }

    for i in range(1, 29):
        data[f'V{i}'] = np.random.randn(n_samples)

    return pd.DataFrame(data)


@pytest.fixture
def sample_engineered_data(sample_raw_data):
    from src.data_processing import feature_engineer
    return feature_engineer(sample_raw_data.copy())
