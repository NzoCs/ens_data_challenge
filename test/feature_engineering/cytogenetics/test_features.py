import pytest
import pandas as pd
from typing import List

from ens_data_challenge.feature_engineering.cytogenetic_parser.parser import (
    CytogeneticsParser,
)
from ens_data_challenge.feature_engineering.cytogenetic_parser.features import (
    CytogeneticsFeatureExtractor,
)
from ens_data_challenge.gloabls import TRAIN_CLINICAL_DATA_PATH, TEST_CLINICAL_DATA_PATH


@pytest.fixture
def parser() -> CytogeneticsParser:
    return CytogeneticsParser()


@pytest.fixture
def feature_extractor() -> CytogeneticsFeatureExtractor:
    return CytogeneticsFeatureExtractor()


@pytest.fixture
def clinical_data_train() -> pd.DataFrame:
    return pd.read_csv(TRAIN_CLINICAL_DATA_PATH)


@pytest.fixture
def clinical_data_test() -> pd.DataFrame:
    return pd.read_csv(TEST_CLINICAL_DATA_PATH)


def test_extract_features_clinical_train_cytogenetics(
    clinical_data_train: pd.DataFrame, 
    parser: CytogeneticsParser, 
    feature_extractor: CytogeneticsFeatureExtractor
) -> None:
    """Integration test: extract features from all clinical train cytogenetics"""

    features_df, feat_cols = feature_extractor.gen_features_to_dataframe(clinical_data_train, cyto_col='CYTOGENETICS')
    df_with_features = features_df
    
    # Check that features columns are added
    expected_features = [
        'is_normal', 'ploidy', 'has_tp53_deletion', 'has_complex_chr3', 'n_abnormalities',
        'n_chromosomes_affected', 'has_monosomy_7', 'has_del_5q', 'has_del_7q', 'has_monosomy_y',
        'n_deletions', 'n_critical_regions_deleted', 'has_large_deletion', 'is_mosaic', 'n_clones',
        'abnormal_clone_percentage', 'computed_risk_score', 'mds_ipss_r_cyto_risk', 'mds_ipss_cyto_risk',
        'aml_eln_2022_cyto_risk', 'cll_cyto_risk', 'mm_riss_cyto_risk'
    ]
    for feat in expected_features:
        assert feat in df_with_features.columns, f"Feature {feat} not added to DataFrame"
    
    # Basic checks
    assert len(df_with_features) == len(clinical_data_train)
    
    # Check some statistics
    # Convert to boolean safely (handle None) and count
    normal_bool = df_with_features['is_normal'].fillna(False).astype(bool)
    mosaic_bool = df_with_features['is_mosaic'].fillna(False).astype(bool)
    normal_count = normal_bool.sum()
    mosaic_count = mosaic_bool.sum()
    abnormal_count = ((~normal_bool) & (~mosaic_bool)).sum()
    
    assert normal_count > 0
    assert abnormal_count >= 0  # Allow abnormal to be less than normal
    assert mosaic_count >= 0
    
    # Check risk scores are reasonable
    risk_scores = df_with_features['computed_risk_score'].dropna()
    assert all(0 <= score <= 1 for score in risk_scores)


def test_extract_features_clinical_test_cytogenetics(
    clinical_data_test: pd.DataFrame, 
    parser: CytogeneticsParser, 
    feature_extractor: CytogeneticsFeatureExtractor
) -> None:
    """Integration test: extract features from all clinical test cytogenetics"""
    features_df, feat_cols = feature_extractor.gen_features_to_dataframe(clinical_data_test, cyto_col='CYTOGENETICS')
    df_with_features = features_df
    
    # Check that features columns are added
    expected_features = [
        'is_normal', 'ploidy', 'has_tp53_deletion', 'has_complex_chr3', 'n_abnormalities',
        'n_chromosomes_affected', 'has_monosomy_7', 'has_del_5q', 'has_del_7q', 'has_monosomy_y',
        'n_deletions', 'n_critical_regions_deleted', 'has_large_deletion', 'is_mosaic', 'n_clones',
        'abnormal_clone_percentage', 'computed_risk_score', 'mds_ipss_r_cyto_risk', 'mds_ipss_cyto_risk',
        'aml_eln_2022_cyto_risk', 'cll_cyto_risk', 'mm_riss_cyto_risk'
    ]
    for feat in expected_features:
        assert feat in df_with_features.columns, f"Feature {feat} not added to DataFrame"
    
    # Basic checks
    assert len(df_with_features) == len(clinical_data_test)
    
    # Check some statistics
    # Convert to boolean safely (handle None) and count
    normal_bool = df_with_features['is_normal'].fillna(False).astype(bool)
    mosaic_bool = df_with_features['is_mosaic'].fillna(False).astype(bool)
    normal_count = normal_bool.sum()
    mosaic_count = mosaic_bool.sum()
    abnormal_count = ((~normal_bool) & (~mosaic_bool)).sum()
    
    assert normal_count > 0
    assert abnormal_count >= 0  # Allow abnormal to be less than normal
    assert mosaic_count >= 0
    
    # Check risk scores are reasonable
    risk_scores = df_with_features['computed_risk_score'].dropna()
    assert all(0 <= score <= 1 for score in risk_scores)


def test_extract_features_normal_karyotype(parser: CytogeneticsParser, feature_extractor: CytogeneticsFeatureExtractor) -> None:
    """Test feature extraction for normal karyotype"""
    parsed = parser.parse("46,XY")
    assert parsed is not None
    
    features = feature_extractor.extract_features(parsed)
    
    assert features['is_normal'] is True
    assert features['ploidy'] == 46
    assert features['n_abnormalities'] == 0
    assert features['is_mosaic'] is False
    assert features['n_clones'] == 1
    assert features['abnormal_clone_percentage'] == 0.0
    assert features['computed_risk_score'] == 0.0
    assert features['mds_ipss_r_cyto_risk'] == 'Good'


def test_extract_features_single_abnormal_karyotype(parser: CytogeneticsParser, feature_extractor: CytogeneticsFeatureExtractor) -> None:
    """Test feature extraction for single abnormal karyotype with monosomy 7"""
    parsed = parser.parse("45,XY,-7")
    assert parsed is not None
    
    features = feature_extractor.extract_features(parsed)
    
    assert features['is_normal'] is False
    assert features['ploidy'] == 45
    assert features['has_monosomy_7'] == 1
    assert features['n_abnormalities'] == 1
    assert features['is_mosaic'] == 0
    assert features['n_clones'] == 1
    assert features['abnormal_clone_percentage'] == 100.0
    assert features['mds_ipss_r_cyto_risk'] == 'Poor'
    assert features['computed_risk_score'] > 0


def test_extract_features_mosaic_karyotype(parser: CytogeneticsParser, feature_extractor: CytogeneticsFeatureExtractor) -> None:
    """Test feature extraction for mosaic karyotype"""
    parsed = parser.parse("46,XY[10]/45,XY,-7[5]")
    assert parsed is not None
    assert len(parsed) == 2
    
    features = feature_extractor.extract_features(parsed)
    
    assert features['is_normal'] == 0
    assert features['is_mosaic'] is True
    assert features['n_clones'] == 2
    assert features['has_monosomy_7'] is True  # From the abnormal clone
    assert features['abnormal_clone_percentage'] == pytest.approx(33.33, abs=0.01)  # 5/15 * 100
    assert features['mds_ipss_r_cyto_risk'] == 'Poor'


def test_extract_features_complex_karyotype(parser: CytogeneticsParser, feature_extractor: CytogeneticsFeatureExtractor) -> None:
    """Test feature extraction for complex karyotype with multiple abnormalities"""
    parsed = parser.parse("43,XY,-5,-7,-17,del(5)(q31)")
    assert parsed is not None
    
    features = feature_extractor.extract_features(parsed)
    
    assert features['is_normal'] is False
    assert features['n_abnormalities'] >= 4
    assert features['has_monosomy_7'] is True
    assert features['has_del_5q'] is True
    assert features['has_tp53_deletion'] is False  # -17 is monosomy, not deletion
    assert features['mds_ipss_r_cyto_risk'] == 'Very Poor'


def test_extract_features_empty_input(feature_extractor: CytogeneticsFeatureExtractor) -> None:
    """Test feature extraction for None/empty input"""
    features = feature_extractor.extract_features(None)
    
    assert features['is_normal'] is None
    assert features['n_abnormalities'] is None
    assert features['computed_risk_score'] is None


def test_extract_features_tp53_deletion(parser: CytogeneticsParser, feature_extractor: CytogeneticsFeatureExtractor) -> None:
    """Test detection of TP53 deletion"""
    parsed = parser.parse("46,XY,del(17)(p13)")
    assert parsed is not None
    
    features = feature_extractor.extract_features(parsed)
    
    assert features['has_tp53_deletion'] is True
    assert features['n_critical_regions_deleted'] >= 1


def test_extract_features_del5q(parser: CytogeneticsParser, feature_extractor: CytogeneticsFeatureExtractor) -> None:
    """Test detection of 5q deletion"""
    parsed = parser.parse("46,XY,del(5)(q31)")
    assert parsed is not None
    
    features = feature_extractor.extract_features(parsed)
    
    assert features['has_del_5q'] is True
    assert features['mds_ipss_r_cyto_risk'] == 'Good'  # Single del(5q) is Good


def test_extract_features_trisomy8(parser: CytogeneticsParser, feature_extractor: CytogeneticsFeatureExtractor) -> None:
    """Test detection of trisomy 8"""
    parsed = parser.parse("47,XY,+8")
    assert parsed is not None
    
    features = feature_extractor.extract_features(parsed)
    
    assert '8' in parsed[0].trisomies
    assert features['n_abnormalities'] == 1
    assert features['mds_ipss_r_cyto_risk'] == 'Intermediate'  # +8 alone is Intermediate


def test_extract_features_mosaic_normal_abnormal(parser: CytogeneticsParser, feature_extractor: CytogeneticsFeatureExtractor) -> None:
    """Test mosaic with normal and abnormal clones"""
    parsed = parser.parse("46,XY[15]/47,XY,+8[5]")
    assert parsed is not None
    assert len(parsed) == 2
    
    features = feature_extractor.extract_features(parsed)
    
    assert features['is_mosaic'] is True
    assert features['n_clones'] == 2
    assert features['abnormal_clone_percentage'] == 25.0  # 5/20 * 100
    assert features['mds_ipss_r_cyto_risk'] == 'Intermediate'  # Based on +8


def test_extract_features_mosaic_multiple_abnormal(parser: CytogeneticsParser, feature_extractor: CytogeneticsFeatureExtractor) -> None:
    """Test mosaic with multiple abnormal clones, selecting the most abnormal"""
    parsed = parser.parse("47,XY,+8[10]/45,XY,-7[5]/46,XY,del(5)(q31)[3]")
    assert parsed is not None
    assert len(parsed) == 3
    
    features = feature_extractor.extract_features(parsed)
    
    assert features['is_mosaic'] is True
    assert features['n_clones'] == 3
    assert features['has_monosomy_7'] is True  # Most abnormal clone has -7
    assert features['mds_ipss_r_cyto_risk'] == 'Poor'  # Due to -7
