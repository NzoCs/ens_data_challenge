import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List

from ens_data_challenge.feature_engineering.feature_engineering import FeatureEngineering
from ens_data_challenge.gloabls import (
    TRAIN_CLINICAL_DATA_PATH,
    TEST_CLINICAL_DATA_PATH,
    TRAIN_MOLECULAR_DATA_PATH,
    TEST_MOLECULAR_DATA_PATH,
    TRAIN_TARGET_PATH
)


@pytest.fixture
def clinical_data_train() -> pd.DataFrame:
    """Load clinical train data"""
    return pd.read_csv(TRAIN_CLINICAL_DATA_PATH)


@pytest.fixture
def clinical_data_test() -> pd.DataFrame:
    """Load clinical test data"""
    return pd.read_csv(TEST_CLINICAL_DATA_PATH)


@pytest.fixture
def molecular_data_train() -> pd.DataFrame:
    """Load molecular train data"""
    return pd.read_csv(TRAIN_MOLECULAR_DATA_PATH)


@pytest.fixture
def molecular_data_test() -> pd.DataFrame:
    """Load molecular test data"""
    return pd.read_csv(TEST_MOLECULAR_DATA_PATH)

@pytest.fixture
def target_data() -> pd.DataFrame:
    """Create empty target data DataFrame"""
    return pd.read_csv(TRAIN_TARGET_PATH)


@pytest.fixture
def feature_engineering(
    clinical_data_train: pd.DataFrame, 
    clinical_data_test: pd.DataFrame, 
    molecular_data_train: pd.DataFrame, 
    molecular_data_test: pd.DataFrame,
    target_data: pd.DataFrame
) -> FeatureEngineering:
    """Create FeatureEngineering instance"""
    return FeatureEngineering(
        clinical_data_train, 
        clinical_data_test, 
        molecular_data_train, 
        molecular_data_test, 
        target_data=target_data
    )


class TestFeatureEngineering:
    """Test the FeatureEngineering class"""

    def test_initialization(self, feature_engineering: FeatureEngineering) -> None:
        """Test that FeatureEngineering initializes correctly"""
        assert feature_engineering.X_train is not None
        assert feature_engineering.X_test is not None
        assert feature_engineering.molecular_data_train is not None
        assert feature_engineering.molecular_data_test is not None
        
        # X_train and X_test should be empty DataFrames initially
        assert len(feature_engineering.get_X_train()) == 0
        assert len(feature_engineering.get_X_test()) == 0

    def test_create_cytogenetics_features(self, feature_engineering: FeatureEngineering) -> None:
        """Test cytogenetics feature creation"""
        # Create cytogenetics features
        new_features: List[str] = feature_engineering.create_cytogenetics_features()
        
        # Check that features were created
        assert len(new_features) > 0
        assert isinstance(new_features, list)
        
        # Check that X_train and X_test have the features
        X_train: pd.DataFrame = feature_engineering.get_X_train()
        X_test: pd.DataFrame = feature_engineering.get_X_test()
        
        assert len(X_train) > 0
        assert len(X_test) > 0
        assert X_train.shape[1] == len(new_features)
        assert X_test.shape[1] == len(new_features)
        
        # Check some expected features
        expected_features = [
            'is_normal', 'ploidy', 'has_tp53_deletion', 'has_complex_chr3', 
            'n_abnormalities', 'n_chromosomes_affected', 'computed_risk_score'
        ]
        for feat in expected_features:
            assert feat in new_features, f"Expected feature {feat} not found"
            assert feat in X_train.columns
            assert feat in X_test.columns

    def test_create_ratios_and_interactions(self, feature_engineering: FeatureEngineering) -> None:
        """Test ratios and interactions feature creation"""
        # First create cytogenetics features (required for ratios)
        feature_engineering.create_cytogenetics_features()
        
        # Create ratios and interactions
        ratio_features: List[str] = feature_engineering.create_ratios_and_interactions()
        
        # Check that features were created
        assert len(ratio_features) == 4  # wbc_anc_ratio, plt_hb_ratio, blast_cyto_complexity, tumor_burden_composite
        assert 'wbc_anc_ratio' in ratio_features
        assert 'plt_hb_ratio' in ratio_features
        assert 'blast_cyto_complexity' in ratio_features
        assert 'tumor_burden_composite' in ratio_features
        
        # Check that X_train and X_test have the features
        X_train: pd.DataFrame = feature_engineering.get_X_train()
        X_test: pd.DataFrame = feature_engineering.get_X_test()
        
        for feat in ratio_features:
            assert feat in X_train.columns
            assert feat in X_test.columns

    def test_create_severity_scores(self, feature_engineering: FeatureEngineering) -> None:
        """Test severity scores feature creation"""
        # Create severity scores
        severity_features: List[str] = feature_engineering.create_severity_scores()
        
        # Check that features were created
        assert len(severity_features) == 1
        assert 'cytopenias_count' in severity_features
        
        # Check that X_train and X_test have the features
        X_train: pd.DataFrame = feature_engineering.get_X_train()
        X_test: pd.DataFrame = feature_engineering.get_X_test()
        
        assert 'cytopenias_count' in X_train.columns
        assert 'cytopenias_count' in X_test.columns
        
        # Check that cytopenias_count is integer between 0 and 3
        assert X_train['cytopenias_count'].dtype in [np.int32, np.int64, int]
        assert X_test['cytopenias_count'].dtype in [np.int32, np.int64, int]
        assert all(0 <= val <= 3 for val in X_train['cytopenias_count'].dropna())
        assert all(0 <= val <= 3 for val in X_test['cytopenias_count'].dropna())

    def test_full_pipeline(self, feature_engineering: FeatureEngineering) -> None:
        """Test the complete feature engineering pipeline"""
        # Run all feature creation methods
        cyto_features: List[str] = feature_engineering.create_cytogenetics_features()
        ratio_features: List[str] = feature_engineering.create_ratios_and_interactions()
        severity_features: List[str] = feature_engineering.create_severity_scores()
        
        # Check total features
        X_train: pd.DataFrame = feature_engineering.get_X_train()
        X_test: pd.DataFrame = feature_engineering.get_X_test()
        
        total_features = len(cyto_features) + len(ratio_features) + len(severity_features)
        assert X_train.shape[1] == total_features
        assert X_test.shape[1] == total_features
        
        # Check that no NaN values in final features (except where expected)
        # Cytogenetics features should not have NaN for parsed karyotypes
        cyto_cols = [col for col in X_train.columns if col in cyto_features]
        for col in cyto_cols:
            if col != 'computed_risk_score':  # risk score can be None for failed parsing
                non_null_ratio = X_train[col].notna().mean()
                assert non_null_ratio > 0.8, f"Too many NaN in {col}: {non_null_ratio:.2%}"

    def test_idempotency_cytogenetics(self, feature_engineering: FeatureEngineering) -> None:
        """Test that cytogenetics features are idempotent"""
        # Create features first time
        features_1 = feature_engineering.create_cytogenetics_features()
        X_train_1 = feature_engineering.get_X_train().copy()
        X_test_1 = feature_engineering.get_X_test().copy()
        
        # Create features second time
        features_2 = feature_engineering.create_cytogenetics_features()
        X_train_2 = feature_engineering.get_X_train().copy()
        X_test_2 = feature_engineering.get_X_test().copy()
        
        # Should be identical
        assert features_1 == features_2
        pd.testing.assert_frame_equal(X_train_1, X_train_2)
        pd.testing.assert_frame_equal(X_test_1, X_test_2)

    def test_idempotency_ratios(self, feature_engineering: FeatureEngineering) -> None:
        """Test that ratios and interactions are idempotent"""
        # Create cytogenetics first
        feature_engineering.create_cytogenetics_features()
        
        # Create ratios first time
        ratios_1 = feature_engineering.create_ratios_and_interactions()
        X_train_1 = feature_engineering.get_X_train().copy()
        X_test_1 = feature_engineering.get_X_test().copy()
        
        # Create ratios second time
        ratios_2 = feature_engineering.create_ratios_and_interactions()
        X_train_2 = feature_engineering.get_X_train().copy()
        X_test_2 = feature_engineering.get_X_test().copy()
        
        # Should be identical
        assert ratios_1 == ratios_2
        pd.testing.assert_frame_equal(X_train_1, X_train_2)
        pd.testing.assert_frame_equal(X_test_1, X_test_2)

    def test_idempotency_severity(self, feature_engineering: FeatureEngineering) -> None:
        """Test that severity scores are idempotent"""
        # Create severity first time
        severity_1 = feature_engineering.create_severity_scores()
        X_train_1 = feature_engineering.get_X_train().copy()
        X_test_1 = feature_engineering.get_X_test().copy()
        
        # Create severity second time
        severity_2 = feature_engineering.create_severity_scores()
        X_train_2 = feature_engineering.get_X_train().copy()
        X_test_2 = feature_engineering.get_X_test().copy()
        
        # Should be identical
        assert severity_1 == severity_2
        pd.testing.assert_frame_equal(X_train_1, X_train_2)
        pd.testing.assert_frame_equal(X_test_1, X_test_2)
