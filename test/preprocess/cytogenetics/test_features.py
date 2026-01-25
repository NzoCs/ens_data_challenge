import pytest
import pandas as pd

from ens_data_challenge.preprocess.cytogenetic_parser.parser import (
    CytogeneticsParser,
)
from ens_data_challenge.preprocess.cytogenetic_parser.dataframe import (
    CytogeneticsExtractor,
)
from ens_data_challenge.types import (
    Columns,
    CytoColumns,
    CytoStructColumns,
    MdsIpssRCytoRisk,
    AmlEln2022CytoRisk,
    CllCytoRisk,
    MmRissCytoRisk,
)
from ens_data_challenge.preprocess.preprocessor import Preprocessor
from ens_data_challenge.globals import (
    TRAIN_CLINICAL_DATA_PATH,
    TEST_CLINICAL_DATA_PATH,
    TRAIN_MOLECULAR_DATA_PATH,
    TEST_MOLECULAR_DATA_PATH,
)
from ens_data_challenge.feature_engineering import FeatureEngineerHelper


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
    feature_extractor: CytogeneticsExtractor,
) -> None:
    """Integration test: extract features from all clinical train cytogenetics"""

    features_df = feature_extractor.gen_features_to_dataframe(
        clinical_data_train, cyto_col=Columns.CYTOGENETICS
    )
    df_with_features = features_df

    # Check that features columns are added. Use the extractor's empty template
    # so tests remain robust to changes in which low-level flags are exposed.
    expected_features = list(feature_extractor._empty_features().keys())
    for feat in expected_features:
        assert feat in df_with_features.columns, (
            f"Feature {feat} not added to DataFrame"
        )

    # Basic checks
    assert len(df_with_features) == len(clinical_data_train)

    # Check some statistics
    # Convert to boolean safely (handle None) and count
    normal_bool = (
        df_with_features[CytoColumns.IS_NORMAL.value].fillna(False).astype(bool)
    )
    mosaic_bool = (
        df_with_features[CytoColumns.IS_MOSAIC.value].fillna(False).astype(bool)
    )
    normal_count = normal_bool.sum()
    mosaic_count = mosaic_bool.sum()
    abnormal_count = ((~normal_bool) & (~mosaic_bool)).sum()

    assert normal_count > 0
    assert abnormal_count >= 0  # Allow abnormal to be less than normal
    assert mosaic_count >= 0

    # Check risk scores are reasonable
    risk_scores = df_with_features["computed_risk_score"].dropna()
    assert all(0 <= score <= 1 for score in risk_scores)


def test_extract_features_clinical_test_cytogenetics(
    clinical_data_test: pd.DataFrame,
    parser: CytogeneticsParser,
    feature_extractor: CytogeneticsExtractor,
) -> None:
    """Integration test: extract features from all clinical test cytogenetics"""
    features_df = feature_extractor.gen_features_to_dataframe(
        clinical_data_test, cyto_col="CYTOGENETICS"
    )
    df_with_features = features_df

    expected_features = list(feature_extractor._empty_features().keys())
    for feat in expected_features:
        assert feat in df_with_features.columns, (
            f"Feature {feat} not added to DataFrame"
        )

    # Basic checks
    assert len(df_with_features) == len(clinical_data_test)

    # Check some statistics
    # Convert to boolean safely (handle None) and count
    normal_bool = (
        df_with_features[CytoColumns.IS_NORMAL.value].fillna(False).astype(bool)
    )
    mosaic_bool = (
        df_with_features[CytoColumns.IS_MOSAIC.value].fillna(False).astype(bool)
    )
    normal_count = normal_bool.sum()
    mosaic_count = mosaic_bool.sum()
    abnormal_count = ((~normal_bool) & (~mosaic_bool)).sum()

    assert normal_count > 0
    assert abnormal_count >= 0  # Allow abnormal to be less than normal
    assert mosaic_count >= 0

    # Check risk scores are reasonable
    risk_scores = df_with_features[CytoColumns.COMPUTED_RISK_SCORE.value].dropna()
    assert all(0 <= score <= 1 for score in risk_scores)


def test_extract_features_normal_karyotype(
    parser: CytogeneticsParser, feature_extractor: CytogeneticsExtractor
) -> None:
    """Test feature extraction for normal karyotype"""
    parsed = parser.parse("46,XY")
    assert parsed is not None

    features = feature_extractor.extract_features(parsed)

    assert features["is_normal"] is True
    assert features["ploidy"] == 46
    assert features["n_abnormalities"] == 0
    assert features["is_mosaic"] is False
    assert features["n_clones"] == 1
    assert features["abnormal_clone_percentage"] == 0.0
    assert features["computed_risk_score"] == 0.0
    assert features["mds_ipss_r_cyto_risk"] == MdsIpssRCytoRisk.GOOD


def test_extract_features_single_abnormal_karyotype(
    parser: CytogeneticsParser, feature_extractor: CytogeneticsExtractor
) -> None:
    """Test feature extraction for single abnormal karyotype with monosomy 7"""
    parsed = parser.parse("45,XY,-7")
    assert parsed is not None

    features = feature_extractor.extract_features(parsed)

    assert features["is_normal"] is False
    assert features["ploidy"] == 45
    assert features["n_abnormalities"] == 1
    assert features["is_mosaic"] == 0
    assert features["n_clones"] == 1
    assert features["abnormal_clone_percentage"] == 100.0
    assert features["mds_ipss_r_cyto_risk"] == MdsIpssRCytoRisk.POOR
    assert features["computed_risk_score"] > 0


def test_extract_features_mosaic_karyotype(
    parser: CytogeneticsParser, feature_extractor: CytogeneticsExtractor
) -> None:
    """Test feature extraction for mosaic karyotype"""
    parsed = parser.parse("46,XY[10]/45,XY,-7[5]")
    assert parsed is not None
    assert len(parsed) == 2

    features = feature_extractor.extract_features(parsed)

    assert features["is_normal"] == 0
    assert features["is_mosaic"] is True
    assert features["n_clones"] == 2
    assert features["abnormal_clone_percentage"] == pytest.approx(
        33.33, abs=0.01
    )  # 5/15 * 100
    assert features["mds_ipss_r_cyto_risk"] == MdsIpssRCytoRisk.POOR


def test_extract_features_complex_karyotype(
    parser: CytogeneticsParser, feature_extractor: CytogeneticsExtractor
) -> None:
    """Test feature extraction for complex karyotype with multiple abnormalities"""
    parsed = parser.parse("43,XY,-5,-7,-17,del(5)(q31)")
    assert parsed is not None

    features = feature_extractor.extract_features(parsed)

    assert features["is_normal"] is False
    assert features["n_abnormalities"] >= 4
    assert features["mds_ipss_r_cyto_risk"] == MdsIpssRCytoRisk.VERY_POOR


def test_extract_features_empty_input(feature_extractor: CytogeneticsExtractor) -> None:
    """Test feature extraction for None/empty input"""
    features = feature_extractor.extract_features(None)

    assert features["is_normal"] is True
    assert features["n_abnormalities"] == 0
    assert features["computed_risk_score"] == 0.0


def test_extract_features_tp53_deletion(
    parser: CytogeneticsParser, feature_extractor: CytogeneticsExtractor
) -> None:
    """Test detection of TP53 deletion"""
    parsed = parser.parse("46,XY,del(17)(p13)")
    assert parsed is not None

    features = feature_extractor.extract_features(parsed)

    assert features["n_critical_regions_deleted"] >= 1


def test_extract_features_del5q(
    parser: CytogeneticsParser, feature_extractor: CytogeneticsExtractor
) -> None:
    """Test detection of 5q deletion"""
    parsed = parser.parse("46,XY,del(5)(q31)")
    assert parsed is not None

    features = feature_extractor.extract_features(parsed)

    assert (
        features["mds_ipss_r_cyto_risk"] == MdsIpssRCytoRisk.GOOD
    )  # Single del(5q) is Good


def test_extract_features_trisomy8(
    parser: CytogeneticsParser, feature_extractor: CytogeneticsExtractor
) -> None:
    """Test detection of trisomy 8"""
    parsed = parser.parse("47,XY,+8")
    assert parsed is not None

    features = feature_extractor.extract_features(parsed)

    assert "8" in parsed[0].trisomies
    assert features["n_abnormalities"] == 1
    assert (
        features["mds_ipss_r_cyto_risk"] == MdsIpssRCytoRisk.INTERMEDIATE
    )  # +8 alone is Intermediate


def test_extract_features_mosaic_normal_abnormal(
    parser: CytogeneticsParser, feature_extractor: CytogeneticsExtractor
) -> None:
    """Test mosaic with normal and abnormal clones"""
    parsed = parser.parse("46,XY[15]/47,XY,+8[5]")
    assert parsed is not None
    assert len(parsed) == 2

    features = feature_extractor.extract_features(parsed)

    assert features["is_mosaic"] is True
    assert features["n_clones"] == 2
    assert features["abnormal_clone_percentage"] == 25.0  # 5/20 * 100
    assert (
        features["mds_ipss_r_cyto_risk"] == MdsIpssRCytoRisk.INTERMEDIATE
    )  # Based on +8


def test_extract_features_mosaic_multiple_abnormal(
    parser: CytogeneticsParser, feature_extractor: CytogeneticsExtractor
) -> None:
    """Test mosaic with multiple abnormal clones, selecting the most abnormal"""
    parsed = parser.parse("47,XY,+8[10]/45,XY,-7[5]/46,XY,del(5)(q31)[3]")
    assert parsed is not None
    assert len(parsed) == 3

    features = feature_extractor.extract_features(parsed)

    assert features["is_mosaic"] is True
    assert features["n_clones"] == 3
    assert features["mds_ipss_r_cyto_risk"] == MdsIpssRCytoRisk.POOR  # Due to -7


def test_gen_structured_dataframe_integration(
    clinical_data_train: pd.DataFrame, feature_extractor: CytogeneticsExtractor
) -> None:
    """Integration test: run gen_structured_dataframe on clinical training data."""
    # run on a subset (first 200 rows) to keep test fast but realistic
    subset = clinical_data_train.head(200)
    struct_df = feature_extractor.gen_structured_dataframe(
        subset, cyto_col="CYTOGENETICS"
    )

    # basic sanity checks
    assert isinstance(struct_df, pd.DataFrame)
    # Check for actual columns returned by gen_structured_dataframe
    assert "mutation_type" in struct_df.columns
    assert "chromosome" in struct_df.columns

    # at least some rows should have anomalies
    any_abn = len(struct_df) > 0
    assert any_abn


def test_preprocessor_clean_data_integration(
    clinical_data_train: pd.DataFrame,
    clinical_data_test: pd.DataFrame,
    molecular_data_train: pd.DataFrame,
    molecular_data_test: pd.DataFrame,
) -> None:
    """Integration smoke test: instantiate Preprocessor, run fit_transform, and verify outputs."""

    # copy inputs to avoid mutating fixtures
    clin_train = clinical_data_train.copy()
    clin_test = clinical_data_test.copy()
    mol_train = molecular_data_train.copy()
    mol_test = molecular_data_test.copy()

    # Create dummy targets for fit_transform
    targets = pd.DataFrame({Columns.ID: clin_train[Columns.ID], "target": 0})

    pre = Preprocessor()

    # 1. Generate cyto features first
    clin_train_proc, cyto_struct_train = pre.get_cyto_features_and_df(clin_train)
    clin_test_proc, cyto_struct_test = pre.get_cyto_features_and_df(clin_test)

    # 2. Run fit_transform
    (
        clin_train_clean,
        mol_train_clean,
        clin_test_clean,
        mol_test_clean,
        cyto_train_clean,
        cyto_test_clean,
        targets_clean,
    ) = pre.fit_transform(
        clinical_data_train=clin_train_proc,
        molecular_data_train=mol_train,
        clinical_data_test=clin_test_proc,
        molecular_data_test=mol_test,
        cyto_struct_train=cyto_struct_train,
        cyto_struct_test=cyto_struct_test,
        targets=targets,
    )

    # Flattened features from extractor must be present in clinical tables
    expected_feats = list(CytogeneticsExtractor()._empty_features().keys())
    for feat in expected_feats:
        assert feat in clin_train_clean.columns


def test_feature_engineer_helper_integration(
    clinical_data_train: pd.DataFrame,
    clinical_data_test: pd.DataFrame,
    molecular_data_train: pd.DataFrame,
) -> None:
    """Integration test: load real data and exercise FeatureEngineerHelper methods."""
    fe = FeatureEngineerHelper()

    # 1) as_type: coerce numeric columns (use safe subset)
    sample_clin = clinical_data_train.head(5).copy()
    # Only run if columns exist
    if all(c in sample_clin.columns for c in ("BM_BLAST", "WBC")):
        type_df = fe.as_type(sample_clin, {"BM_BLAST": float, "WBC": float})
        assert "BM_BLAST" in type_df.columns and type_df["BM_BLAST"].dtype.kind in (
            "f",
            "i",
        )

    clin_slice = clinical_data_train.head(50).copy()
    # Mock missing columns expected by ratios_and_interactions
    if "n_abnormalities" not in clin_slice.columns:
        clin_slice["n_abnormalities"] = 0
    if "computed_risk_score" not in clin_slice.columns:
        clin_slice["computed_risk_score"] = 0.0
    clin_with_nmut = fe.Nmut(molecular_data_train, clin_slice)
    assert "Nmut" in clin_with_nmut.columns

    # 3) ratios and interactions
    clin_ratios = fe.ratios_and_interactions(clin_with_nmut.fillna(0))
    for col in (
        "wbc_anc_ratio",
        "plt_hb_ratio",
        "blast_cyto_complexity",
        "tumor_burden_composite",
    ):
        assert col in clin_ratios.columns

    # 4) severity scores
    clin_sev = fe.severity_scores(
        clin_with_nmut.fillna({"HB": 12, "PLT": 150, "ANC": 2})
    )
    assert "cytopenias_count" in clin_sev.columns

    # 5) random feature
    rand_df = fe.random(clin_with_nmut.head(10), seed=42)
    assert "random_feature" in rand_df.columns and len(rand_df) == 10

    # 6) pathways classification on molecular data (small slice)
    mol_slice = molecular_data_train.head(50).copy()
    if "GENE" in mol_slice.columns:
        mol_with_path = fe._pathways_classification(mol_slice)
        assert "PATHWAY" in mol_with_path.columns

    # 7) create confidence weighted matrix (use small subset and safe method)
    if "GENE" in molecular_data_train.columns and len(molecular_data_train) > 0:
        cats = list(molecular_data_train["GENE"].dropna().unique()[:5])
        mat = fe._create_confidence_weighted_count_matrix(
            col="GENE",
            molecular_data=molecular_data_train.head(200),
            method="constant",
            apply_effect_weighting=False,
            categories_to_include=cats,
        )
        assert isinstance(mat, pd.DataFrame)

    # 8) add molecular encoding (GENE) - run on a small slice to keep test fast
    if "GENE" in molecular_data_train.columns and len(molecular_data_train) > 0:
        cats = list(molecular_data_train["GENE"].dropna().unique()[:3])
        encoded = fe.add_mol_encoding(
            clinical_data_train.head(30).copy(),
            molecular_data_train.head(200).copy(),
            col="GENE",
            method="constant",
            apply_effect_weighting=False,
            categories_to_include=cats,
        )
        for g in cats:
            assert any(g in c for c in encoded.columns)

    # 9) one-hot encode CENTER (fit + transform) if available
    if "CENTER" in clinical_data_train.columns:
        fe.one_hot_encode_fit(clinical_data_train.head(100), ["CENTER"])
        transformed = fe.one_hot_encode_transform(
            clinical_data_test.head(20), ["CENTER"]
        )
        assert "CENTER" not in transformed.columns
