import pytest
import pandas as pd
import numpy as np
from ens_data_challenge.preprocess import Preprocessor2
from ens_data_challenge.globals import TRAIN_CLINICAL_DATA_PATH, TEST_CLINICAL_DATA_PATH, TRAIN_MOLECULAR_DATA_PATH, TEST_MOLECULAR_DATA_PATH

@pytest.fixture
def sample_data():
    # Load real data but limit to small subset for testing
    clinical_train = pd.read_csv(TRAIN_CLINICAL_DATA_PATH).head(10).copy()
    clinical_test = pd.read_csv(TEST_CLINICAL_DATA_PATH).head(5).copy()
    molecular_train = pd.read_csv(TRAIN_MOLECULAR_DATA_PATH).head(20).copy()
    molecular_test = pd.read_csv(TEST_MOLECULAR_DATA_PATH).head(10).copy()
    
    return clinical_train, clinical_test, molecular_train, molecular_test

def test_preprocessor2_fit(sample_data):
    clinical_train, clinical_test, molecular_train, molecular_test = sample_data
    
    preprocessor = Preprocessor2()
    preprocessor.fit(clinical_train, molecular_train, clinical_test, molecular_test)
    
    # Check imputation values are set
    assert hasattr(preprocessor, 'molecular_impute_values')
    assert 'VAF' in preprocessor.molecular_impute_values
    assert 'GENE' in preprocessor.molecular_impute_values
    
    assert hasattr(preprocessor, 'clinical_impute_values')
    assert 'BM_BLAST' in preprocessor.clinical_impute_values
    assert 'WBC' in preprocessor.clinical_impute_values

def test_preprocessor2_transform(sample_data):
    clinical_train, clinical_test, molecular_train, molecular_test = sample_data
    
    preprocessor = Preprocessor2()
    preprocessor.fit(clinical_train, molecular_train, clinical_test, molecular_test)
    
    result = preprocessor.transform(clinical_train, molecular_train)
    
    # Check structure - result is tuple (clinical, molecular, cyto)
    assert len(result) == 3
    clinical_result, molecular_result, cyto_result = result
    
    # Check imputation applied
    assert not clinical_result['BM_BLAST'].isnull().any()
    assert not clinical_result['WBC'].isnull().any()
    
    # Check cytogenetics features added
    assert 'n_abnormalities' in clinical_result.columns

def test_categorical_consistency(sample_data):
    clinical_train, clinical_test, molecular_train, molecular_test = sample_data
    
    # Modify test to have unknown category not in train/test
    molecular_test.loc[1, 'GENE'] = 'VERY_UNKNOWN_GENE'
    
    preprocessor = Preprocessor2()
    preprocessor.fit(clinical_train, molecular_train, clinical_test, molecular_test)
    
    # Transform test
    result_test = preprocessor.transform(clinical_test, molecular_test)
    clinical_test_result, molecular_test_result, _ = result_test
    
    # Check that unknown gene is replaced with 'OTHER'
    assert 'OTHER' in molecular_test_result['GENE'].values  # type: ignore
    assert 'UNKNOWN_GENE' not in molecular_test_result['GENE'].values  # type: ignore

def test_ploidy_clipping(sample_data):
    clinical_train, clinical_test, molecular_train, molecular_test = sample_data
    
    preprocessor = Preprocessor2()
    preprocessor.fit(clinical_train, molecular_train, clinical_test, molecular_test)
    
    result_train = preprocessor.transform(clinical_train, molecular_train)
    clinical_train_result, _, _ = result_train
    result_test = preprocessor.transform(clinical_test, molecular_test)
    clinical_test_result, _, _ = result_test
    
    # Check ploidy exists and is within range (assuming data has normal ploidy)
    assert 'ploidy' in clinical_train_result.columns
    assert clinical_train_result['ploidy'].max() <= 50  # type: ignore
    assert clinical_test_result['ploidy'].max() <= 50  # type: ignore
    assert clinical_train_result['ploidy'].min() >= 30  # type: ignore
    assert clinical_test_result['ploidy'].min() >= 30  # type: ignore

