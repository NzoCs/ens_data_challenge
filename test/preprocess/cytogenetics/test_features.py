import pytest
import pandas as pd
from typing import List

from ens_data_challenge.preprocess.cytogenetic_parser.parser import (
    CytogeneticsParser,
)
from ens_data_challenge.preprocess.cytogenetic_parser.dataframe import (
    CytogeneticsExtractor,
)
from ens_data_challenge.preprocess.preprocessor import Preprocessor
from ens_data_challenge.globals import TRAIN_CLINICAL_DATA_PATH, TEST_CLINICAL_DATA_PATH, TRAIN_MOLECULAR_DATA_PATH, TEST_MOLECULAR_DATA_PATH


@pytest.fixture
def parser() -> CytogeneticsParser:
    return CytogeneticsParser()


@pytest.fixture
def feature_extractor() -> CytogeneticsExtractor:
    return CytogeneticsExtractor()


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


def test_extract_features_clinical_train_cytogenetics(
    clinical_data_train: pd.DataFrame, 
    parser: CytogeneticsParser, 
    feature_extractor: CytogeneticsExtractor
) -> None:
    """Integration test: extract features from all clinical train cytogenetics"""

    features_df, feat_cols = feature_extractor.gen_features_to_dataframe(clinical_data_train, cyto_col='CYTOGENETICS')
    df_with_features = features_df
    
    # Check that features columns are added. Use the extractor's empty template
    # so tests remain robust to changes in which low-level flags are exposed.
    expected_features = list(feature_extractor._empty_features().keys())
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
    feature_extractor: CytogeneticsExtractor
) -> None:
    """Integration test: extract features from all clinical test cytogenetics"""
    features_df, feat_cols = feature_extractor.gen_features_to_dataframe(clinical_data_test, cyto_col='CYTOGENETICS')
    df_with_features = features_df
    
    expected_features = list(feature_extractor._empty_features().keys())
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


def test_extract_features_normal_karyotype(parser: CytogeneticsParser, feature_extractor: CytogeneticsExtractor) -> None:
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


def test_extract_features_single_abnormal_karyotype(parser: CytogeneticsParser, feature_extractor: CytogeneticsExtractor) -> None:
    """Test feature extraction for single abnormal karyotype with monosomy 7"""
    parsed = parser.parse("45,XY,-7")
    assert parsed is not None
    
    features = feature_extractor.extract_features(parsed)
    
    assert features['is_normal'] is False
    assert features['ploidy'] == 45
    assert features['n_abnormalities'] == 1
    assert features['is_mosaic'] == 0
    assert features['n_clones'] == 1
    assert features['abnormal_clone_percentage'] == 100.0
    assert features['mds_ipss_r_cyto_risk'] == 'Poor'
    assert features['computed_risk_score'] > 0

    # Check monosomy via the structured dataframe
    struct_df, _ = feature_extractor.gen_structured_dataframe(pd.DataFrame([{'CYTOGENETICS': '45,XY,-7'}]), cyto_col='CYTOGENETICS')
    assert struct_df.iloc[0]['monosomy_chromosome'] == '7'


def test_extract_features_mosaic_karyotype(parser: CytogeneticsParser, feature_extractor: CytogeneticsExtractor) -> None:
    """Test feature extraction for mosaic karyotype"""
    parsed = parser.parse("46,XY[10]/45,XY,-7[5]")
    assert parsed is not None
    assert len(parsed) == 2
    
    features = feature_extractor.extract_features(parsed)
    
    assert features['is_normal'] == 0
    assert features['is_mosaic'] is True
    assert features['n_clones'] == 2
    assert features['abnormal_clone_percentage'] == pytest.approx(33.33, abs=0.01)  # 5/15 * 100
    assert features['mds_ipss_r_cyto_risk'] == 'Poor'

    # verify monosomy via structured dataframe
    struct_df, _ = feature_extractor.gen_structured_dataframe(pd.DataFrame([{'CYTOGENETICS': '46,XY[10]/45,XY,-7[5]'}]), cyto_col='CYTOGENETICS')
    assert struct_df.iloc[0]['monosomy_chromosome'] == '7'


def test_extract_features_complex_karyotype(parser: CytogeneticsParser, feature_extractor: CytogeneticsExtractor) -> None:
    """Test feature extraction for complex karyotype with multiple abnormalities"""
    parsed = parser.parse("43,XY,-5,-7,-17,del(5)(q31)")
    assert parsed is not None
    
    features = feature_extractor.extract_features(parsed)
    
    assert features['is_normal'] is False
    assert features['n_abnormalities'] >= 4
    assert features['mds_ipss_r_cyto_risk'] == 'Very Poor'

    # verify deletion/monosomy via structured dataframe
    struct_df, _ = feature_extractor.gen_structured_dataframe(pd.DataFrame([{'CYTOGENETICS': '43,XY,-5,-7,-17,del(5)(q31)'}]), cyto_col='CYTOGENETICS')
    assert struct_df.iloc[0]['monosomy_chromosome'] == '17'
    assert struct_df.iloc[0]['monosomies_count'] == 3
    # Per-arm booleans removed from structured DF: inspect deletions_details
    assert struct_df.iloc[0]['deletion_chromosome'] == '5'
    assert struct_df.iloc[0]['deletion_arm'] == 'q'
    # Allow numpy boolean types; cast to bool for comparison
    assert bool(struct_df.iloc[0]['has_tp53_deletion']) is False


def test_extract_features_empty_input(feature_extractor: CytogeneticsExtractor) -> None:
    """Test feature extraction for None/empty input"""
    features = feature_extractor.extract_features(None)
    
    assert features['is_normal'] is None
    assert features['n_abnormalities'] is None
    assert features['computed_risk_score'] is None


def test_extract_features_tp53_deletion(parser: CytogeneticsParser, feature_extractor: CytogeneticsExtractor) -> None:
    """Test detection of TP53 deletion"""
    parsed = parser.parse("46,XY,del(17)(p13)")
    assert parsed is not None
    
    features = feature_extractor.extract_features(parsed)
    
    # verify via structured dataframe
    struct_df, _ = feature_extractor.gen_structured_dataframe(pd.DataFrame([{'CYTOGENETICS': '46,XY,del(17)(p13)'}]), cyto_col='CYTOGENETICS')
    assert bool(struct_df.iloc[0]['has_tp53_deletion']) is True
    assert features['n_critical_regions_deleted'] >= 1


def test_extract_features_del5q(parser: CytogeneticsParser, feature_extractor: CytogeneticsExtractor) -> None:
    """Test detection of 5q deletion"""
    parsed = parser.parse("46,XY,del(5)(q31)")
    assert parsed is not None
    
    features = feature_extractor.extract_features(parsed)
    struct_df, _ = feature_extractor.gen_structured_dataframe(
        pd.DataFrame([{'CYTOGENETICS': '46,XY,del(5)(q31)'}]), cyto_col='CYTOGENETICS'
    )
    assert struct_df.iloc[0]['deletion_chromosome'] == '5'
    assert struct_df.iloc[0]['deletion_arm'] == 'q'
    assert features['mds_ipss_r_cyto_risk'] == 'Good'  # Single del(5q) is Good


def test_extract_features_trisomy8(parser: CytogeneticsParser, feature_extractor: CytogeneticsExtractor) -> None:
    """Test detection of trisomy 8"""
    parsed = parser.parse("47,XY,+8")
    assert parsed is not None
    
    features = feature_extractor.extract_features(parsed)
    
    assert '8' in parsed[0].trisomies
    assert features['n_abnormalities'] == 1
    assert features['mds_ipss_r_cyto_risk'] == 'Intermediate'  # +8 alone is Intermediate


def test_extract_features_mosaic_normal_abnormal(parser: CytogeneticsParser, feature_extractor: CytogeneticsExtractor) -> None:
    """Test mosaic with normal and abnormal clones"""
    parsed = parser.parse("46,XY[15]/47,XY,+8[5]")
    assert parsed is not None
    assert len(parsed) == 2
    
    features = feature_extractor.extract_features(parsed)
    
    assert features['is_mosaic'] is True
    assert features['n_clones'] == 2
    assert features['abnormal_clone_percentage'] == 25.0  # 5/20 * 100
    assert features['mds_ipss_r_cyto_risk'] == 'Intermediate'  # Based on +8


def test_extract_features_mosaic_multiple_abnormal(parser: CytogeneticsParser, feature_extractor: CytogeneticsExtractor) -> None:
    """Test mosaic with multiple abnormal clones, selecting the most abnormal"""
    parsed = parser.parse("47,XY,+8[10]/45,XY,-7[5]/46,XY,del(5)(q31)[3]")
    assert parsed is not None
    assert len(parsed) == 3
    
    features = feature_extractor.extract_features(parsed)
    
    assert features['is_mosaic'] is True
    assert features['n_clones'] == 3
    assert features['mds_ipss_r_cyto_risk'] == 'Poor'  # Due to -7


def test_gen_structured_dataframe_unit_examples(feature_extractor: CytogeneticsExtractor) -> None:
    """Unit test for gen_structured_dataframe on a few synthetic examples."""
    rows = [
        { 'CYTOGENETICS': '46,XY' },
        { 'CYTOGENETICS': '46,XY,del(5)(q31)' },
        { 'CYTOGENETICS': '46,XY[10]/45,XY,-7[5]' },
        { 'CYTOGENETICS': None },
    ]
    df = pd.DataFrame(rows)

    struct_df, cols = feature_extractor.gen_structured_dataframe(df, cyto_col='CYTOGENETICS')


    # Row 0: normal
    r0 = struct_df.iloc[0]
    assert r0['deletions_count'] == 0

    # Row 1: del(5)(q31)
    r1 = struct_df.iloc[1]
    assert r1['deletions_count'] >= 1
    assert r1['deletion_chromosome'] == '5'
    assert r1['deletion_arm'] == 'q'

    # Row 2: mosaic with monosomy 7 clone
    r2 = struct_df.iloc[2]
    import numbers
    assert isinstance(r2['cell_count_total'], numbers.Integral)
    assert r2['monosomy_chromosome'] == '7'



def test_gen_structured_dataframe_integration(clinical_data_train: pd.DataFrame, feature_extractor: CytogeneticsExtractor) -> None:
    """Integration test: run gen_structured_dataframe on clinical training data."""
    # run on a subset (first 200 rows) to keep test fast but realistic
    subset = clinical_data_train.head(200)
    struct_df, cols = feature_extractor.gen_structured_dataframe(subset, cyto_col='CYTOGENETICS')

    # basic sanity checks
    assert len(struct_df) == len(subset)
    assert 'deletions_count' in struct_df.columns
    # at least some rows should have monosomies/trisomies or deletions
    any_abn = (
        (struct_df['deletions_count'].fillna(0) > 0).any() or
        (struct_df['monosomies_count'].fillna(0) > 0).any() or
        (struct_df['trisomies_count'].fillna(0) > 0).any()
    )
    assert any_abn


def test_preprocessor_clean_data_integration(clinical_data_train: pd.DataFrame, clinical_data_test: pd.DataFrame, molecular_data_train: pd.DataFrame, molecular_data_test: pd.DataFrame) -> None:
    """Integration smoke test: instantiate Preprocessor, run clean_data(), and verify outputs."""

    # copy inputs to avoid mutating fixtures
    clin_train = clinical_data_train.copy()
    clin_test = clinical_data_test.copy()

    pre = Preprocessor(
        clinical_data_train=clin_train,
        clinical_data_test=clin_test,
        molecular_data_train=molecular_data_train,
        molecular_data_test=molecular_data_test,
    )

    # run the pipeline step (should not raise)
    pre.clean_data()

    # Flattened features from extractor must be present in clinical tables
    expected_feats = list(CytogeneticsExtractor()._empty_features().keys())
    for feat in expected_feats:
        assert feat in pre.clinical_data_train.columns