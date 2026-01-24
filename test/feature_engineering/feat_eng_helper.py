import pytest
import pandas as pd
import numpy as np

from ens_data_challenge.feature_engineering import FeatureEngineerHelper
from ens_data_challenge.globals import (
    TRAIN_CLINICAL_DATA_PATH,
    TEST_CLINICAL_DATA_PATH,
    TRAIN_MOLECULAR_DATA_PATH,
    TEST_MOLECULAR_DATA_PATH,
    TRAIN_TARGET_PATH,
)


@pytest.fixture
def clinical_data_train() -> pd.DataFrame:
    return pd.read_csv(TRAIN_CLINICAL_DATA_PATH)


@pytest.fixture
def clinical_data_test() -> pd.DataFrame:
    return pd.read_csv(TEST_CLINICAL_DATA_PATH)


@pytest.fixture
def molecular_data_train() -> pd.DataFrame:
    return pd.read_csv(TRAIN_MOLECULAR_DATA_PATH)


@pytest.fixture
def molecular_data_test() -> pd.DataFrame:
    return pd.read_csv(TEST_MOLECULAR_DATA_PATH)


@pytest.fixture
def target_data() -> pd.DataFrame:
    return pd.read_csv(TRAIN_TARGET_PATH)


@pytest.fixture
def helper() -> FeatureEngineerHelper:
    return FeatureEngineerHelper()


def test_as_type(helper: FeatureEngineerHelper):
    df = pd.DataFrame({'a': ['1', '2'], 'b': [1.5, 2.5]})
    out = helper.as_type(df, {'a': int, 'b': float})
    assert out['a'].dtype == int or pd.api.types.is_integer_dtype(out['a'])
    assert pd.api.types.is_float_dtype(out['b'])


def test_encode_risk(helper: FeatureEngineerHelper):
    df = pd.DataFrame({'risk': ['low', 'high', 'very low', 'intermediate']})
    out = helper.encode_risk(df, ['risk'])
    expected = [1, 3, 0, 2]
    assert out['risk'].tolist() == expected


def test_one_hot_fit_transform(helper: FeatureEngineerHelper):
    train = pd.DataFrame({'cat': ['A', 'B', 'A']})
    helper.one_hot_encode_fit(train, ['cat'])
    transformed = helper.one_hot_encode_transform(train, ['cat'])
    # drop='first' => for two categories expect one column
    cols = [c for c in transformed.columns if c.startswith('cat_')]
    assert len(cols) == 1
    assert set(transformed.index) == set(train.index)


def test_Nmut(helper: FeatureEngineerHelper, molecular_data_train: pd.DataFrame, clinical_data_train: pd.DataFrame):
    out = helper.Nmut(molecular_data_train, clinical_data_train)
    # Ensure Nmut column exists and non-negative
    assert 'Nmut' in out.columns
    assert (out['Nmut'] >= 0).all()


def test_ratios_and_interactions(helper: FeatureEngineerHelper, clinical_data_train: pd.DataFrame):
    # use a small sample of the real clinical data
    clinical = clinical_data_train.head(10).copy()
    # ensure required columns exist for the test; skip otherwise
    required = ['WBC', 'ANC', 'PLT', 'HB', 'BM_BLAST', 'n_abnormalities', 'computed_risk_score']
    for col in required:
        if col not in clinical.columns:
            pytest.skip(f"Column {col} not present in clinical data")
    out = helper.ratios_and_interactions(clinical)
    assert 'wbc_anc_ratio' in out.columns
    assert 'plt_hb_ratio' in out.columns
    assert 'blast_cyto_complexity' in out.columns
    assert 'tumor_burden_composite' in out.columns


def test_random_reproducible(helper: FeatureEngineerHelper):
    df = pd.DataFrame({'x': [0, 1, 2]})
    out1 = helper.random(df, seed=42)
    out2 = helper.random(df, seed=42)
    assert 'random_feature' in out1.columns
    assert np.allclose(out1['random_feature'], out2['random_feature'])


def test_severity_scores(helper: FeatureEngineerHelper, clinical_data_train: pd.DataFrame):
    clinical = clinical_data_train.head(10).copy()
    required = ['HB', 'PLT', 'ANC']
    for col in required:
        if col not in clinical.columns:
            pytest.skip(f"Column {col} not present in clinical data")
    out = helper.severity_scores(clinical)
    assert 'cytopenias_count' in out.columns
    vals = out['cytopenias_count'].tolist()
    assert all(0 <= v <= 3 for v in vals)

