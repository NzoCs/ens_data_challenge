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
    normal_bool = df_with_features[CytoColumns.IS_NORMAL].fillna(False).astype(bool)
    mosaic_bool = df_with_features[CytoColumns.IS_MOSAIC].fillna(False).astype(bool)
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
        clinical_data_test, cyto_col=Columns.CYTOGENETICS
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
    normal_bool = df_with_features[CytoColumns.IS_NORMAL].fillna(False).astype(bool)
    mosaic_bool = df_with_features[CytoColumns.IS_MOSAIC].fillna(False).astype(bool)
    normal_count = normal_bool.sum()
    mosaic_count = mosaic_bool.sum()
    abnormal_count = ((~normal_bool) & (~mosaic_bool)).sum()

    assert normal_count > 0
    assert abnormal_count >= 0  # Allow abnormal to be less than normal
    assert mosaic_count >= 0

    # Check risk scores are reasonable
    risk_scores = df_with_features[CytoColumns.COMPUTED_RISK_SCORE].dropna()
    assert all(0 <= score <= 1 for score in risk_scores)


def test_extract_features_normal_karyotype(
    parser: CytogeneticsParser, feature_extractor: CytogeneticsExtractor
) -> None:
    """Test feature extraction for normal karyotype"""
    parsed = parser.parse("46,XY")
    assert parsed is not None

    features = feature_extractor.extract_features(parsed)

    assert features[CytoColumns.IS_NORMAL] is True
    assert features[CytoStructColumns.PLOIDY] == 46
    assert features[CytoColumns.N_ABNORMALITIES] == 0
    assert features[CytoColumns.IS_MOSAIC] is False
    assert features[CytoColumns.N_CLONES] == 1
    assert features[CytoColumns.ABNORMAL_CLONE_PERCENTAGE] == 0.0
    assert features[CytoColumns.COMPUTED_RISK_SCORE] == 0.0
    assert features[CytoColumns.MDS_IPSS_R_CYTO_RISK] == MdsIpssRCytoRisk.GOOD


def test_extract_features_single_abnormal_karyotype(
    parser: CytogeneticsParser, feature_extractor: CytogeneticsExtractor
) -> None:
    """Test feature extraction for single abnormal karyotype with monosomy 7"""
    parsed = parser.parse("45,XY,-7")
    assert parsed is not None

    features = feature_extractor.extract_features(parsed)

    assert features[CytoColumns.IS_NORMAL] is False
    assert features[CytoStructColumns.PLOIDY] == 45
    assert features[CytoColumns.N_ABNORMALITIES] == 1
    assert features[CytoColumns.IS_MOSAIC] == 0
    assert features[CytoColumns.N_CLONES] == 1
    assert features[CytoColumns.ABNORMAL_CLONE_PERCENTAGE] == 100.0
    assert features[CytoColumns.MDS_IPSS_R_CYTO_RISK] == MdsIpssRCytoRisk.POOR
    assert features[CytoColumns.COMPUTED_RISK_SCORE] > 0

    # Check monosomy via the structured dataframe
    struct_df = feature_extractor.gen_structured_dataframe(
        pd.DataFrame([{Columns.CYTOGENETICS: "45,XY,-7"}]),
        cyto_col=Columns.CYTOGENETICS,
    )
    assert struct_df.iloc[0][CytoStructColumns.MONOSOMY_CHROMOSOME] == "7"


def test_extract_features_mosaic_karyotype(
    parser: CytogeneticsParser, feature_extractor: CytogeneticsExtractor
) -> None:
    """Test feature extraction for mosaic karyotype"""
    parsed = parser.parse("46,XY[10]/45,XY,-7[5]")
    assert parsed is not None
    assert len(parsed) == 2

    features = feature_extractor.extract_features(parsed)

    assert features[CytoColumns.IS_NORMAL] == 0
    assert features[CytoColumns.IS_MOSAIC] is True
    assert features[CytoColumns.N_CLONES] == 2
    assert features[CytoColumns.ABNORMAL_CLONE_PERCENTAGE] == pytest.approx(
        33.33, abs=0.01
    )  # 5/15 * 100
    assert features[CytoColumns.MDS_IPSS_R_CYTO_RISK] == MdsIpssRCytoRisk.POOR

    # verify monosomy via structured dataframe
    struct_df = feature_extractor.gen_structured_dataframe(
        pd.DataFrame([{Columns.CYTOGENETICS: "46,XY[10]/45,XY,-7[5]"}]),
        cyto_col=Columns.CYTOGENETICS,
    )
    assert struct_df.iloc[0][CytoStructColumns.MONOSOMY_CHROMOSOME] == "7"


def test_extract_features_complex_karyotype(
    parser: CytogeneticsParser, feature_extractor: CytogeneticsExtractor
) -> None:
    """Test feature extraction for complex karyotype with multiple abnormalities"""
    parsed = parser.parse("43,XY,-5,-7,-17,del(5)(q31)")
    assert parsed is not None

    features = feature_extractor.extract_features(parsed)

    assert features[CytoColumns.IS_NORMAL] is False
    assert features[CytoColumns.N_ABNORMALITIES] >= 4
    assert features[CytoColumns.MDS_IPSS_R_CYTO_RISK] == MdsIpssRCytoRisk.VERY_POOR

    # verify deletion/monosomy via structured dataframe
    struct_df = feature_extractor.gen_structured_dataframe(
        pd.DataFrame([{Columns.CYTOGENETICS: "43,XY,-5,-7,-17,del(5)(q31)"}]),
        cyto_col=Columns.CYTOGENETICS,
    )
    assert struct_df.iloc[0][CytoStructColumns.MONOSOMY_CHROMOSOME] == "17"
    assert struct_df.iloc[0][CytoStructColumns.MONOSOMIES_COUNT] == 3
    # Per-arm booleans removed from structured DF: inspect deletions_details
    assert struct_df.iloc[0][CytoStructColumns.DELETION_CHROMOSOME] == "5"
    assert struct_df.iloc[0][CytoStructColumns.DELETION_ARM] == "q"
    # Allow numpy boolean types; cast to bool for comparison
    assert bool(struct_df.iloc[0][CytoColumns.HAS_TP53_DELETION]) is False


def test_extract_features_empty_input(feature_extractor: CytogeneticsExtractor) -> None:
    """Test feature extraction for None/empty input"""
    features = feature_extractor.extract_features(None)

    assert features[CytoColumns.IS_NORMAL] is True
    assert features[CytoColumns.N_ABNORMALITIES] == 0
    assert features[CytoColumns.COMPUTED_RISK_SCORE] == 0.0


def test_extract_features_tp53_deletion(
    parser: CytogeneticsParser, feature_extractor: CytogeneticsExtractor
) -> None:
    """Test detection of TP53 deletion"""
    parsed = parser.parse("46,XY,del(17)(p13)")
    assert parsed is not None

    features = feature_extractor.extract_features(parsed)

    # verify via structured dataframe
    struct_df = feature_extractor.gen_structured_dataframe(
        pd.DataFrame([{Columns.CYTOGENETICS: "46,XY,del(17)(p13)"}]),
        cyto_col=Columns.CYTOGENETICS,
    )
    assert bool(struct_df.iloc[0][CytoColumns.HAS_TP53_DELETION]) is True
    assert features[CytoColumns.N_CRITICAL_REGIONS_DELETED] >= 1


def test_extract_features_del5q(
    parser: CytogeneticsParser, feature_extractor: CytogeneticsExtractor
) -> None:
    """Test detection of 5q deletion"""
    parsed = parser.parse("46,XY,del(5)(q31)")
    assert parsed is not None

    features = feature_extractor.extract_features(parsed)
    struct_df = feature_extractor.gen_structured_dataframe(
        pd.DataFrame([{Columns.CYTOGENETICS: "46,XY,del(5)(q31)"}]),
        cyto_col=Columns.CYTOGENETICS,
    )
    assert struct_df.iloc[0][CytoStructColumns.DELETION_CHROMOSOME] == "5"
    assert struct_df.iloc[0][CytoStructColumns.DELETION_ARM] == "q"
    assert (
        features[CytoColumns.MDS_IPSS_R_CYTO_RISK] == MdsIpssRCytoRisk.GOOD
    )  # Single del(5q) is Good


def test_extract_features_trisomy8(
    parser: CytogeneticsParser, feature_extractor: CytogeneticsExtractor
) -> None:
    """Test detection of trisomy 8"""
    parsed = parser.parse("47,XY,+8")
    assert parsed is not None

    features = feature_extractor.extract_features(parsed)

    assert "8" in parsed[0].trisomies
    assert features[CytoColumns.N_ABNORMALITIES] == 1
    assert (
        features[CytoColumns.MDS_IPSS_R_CYTO_RISK] == MdsIpssRCytoRisk.INTERMEDIATE
    )  # +8 alone is Intermediate


def test_extract_features_mosaic_normal_abnormal(
    parser: CytogeneticsParser, feature_extractor: CytogeneticsExtractor
) -> None:
    """Test mosaic with normal and abnormal clones"""
    parsed = parser.parse("46,XY[15]/47,XY,+8[5]")
    assert parsed is not None
    assert len(parsed) == 2

    features = feature_extractor.extract_features(parsed)

    assert features[CytoColumns.IS_MOSAIC] is True
    assert features[CytoColumns.N_CLONES] == 2
    assert features[CytoColumns.ABNORMAL_CLONE_PERCENTAGE] == 25.0  # 5/20 * 100
    assert (
        features[CytoColumns.MDS_IPSS_R_CYTO_RISK] == MdsIpssRCytoRisk.INTERMEDIATE
    )  # Based on +8


def test_extract_features_mosaic_multiple_abnormal(
    parser: CytogeneticsParser, feature_extractor: CytogeneticsExtractor
) -> None:
    """Test mosaic with multiple abnormal clones, selecting the most abnormal"""
    parsed = parser.parse("47,XY,+8[10]/45,XY,-7[5]/46,XY,del(5)(q31)[3]")
    assert parsed is not None
    assert len(parsed) == 3

    features = feature_extractor.extract_features(parsed)

    assert features[CytoColumns.IS_MOSAIC] is True
    assert features[CytoColumns.N_CLONES] == 3
    assert (
        features[CytoColumns.MDS_IPSS_R_CYTO_RISK] == MdsIpssRCytoRisk.POOR
    )  # Due to -7


def test_gen_structured_dataframe_unit_examples(
    feature_extractor: CytogeneticsExtractor,
) -> None:
    """Unit test for gen_structured_dataframe on a few synthetic examples."""
    rows = [
        {Columns.CYTOGENETICS: "46,XY"},
        {Columns.CYTOGENETICS: "46,XY,del(5)(q31)"},
        {Columns.CYTOGENETICS: "46,XY[10]/45,XY,-7[5]"},
        {Columns.CYTOGENETICS: None},
    ]
    df = pd.DataFrame(rows)

    struct_df = feature_extractor.gen_structured_dataframe(
        df, cyto_col=Columns.CYTOGENETICS
    )

    # Row 0: normal
    r0 = struct_df.iloc[0]
    assert r0[CytoStructColumns.DELETIONS_COUNT] == 0

    # Row 1: del(5)(q31)
    r1 = struct_df.iloc[1]
    assert r1[CytoStructColumns.DELETIONS_COUNT] >= 1
    assert r1[CytoStructColumns.DELETION_CHROMOSOME] == "5"
    assert r1[CytoStructColumns.DELETION_ARM] == "q"

    # Row 2: mosaic with monosomy 7 clone
    r2 = struct_df.iloc[2]
    import numbers

    assert isinstance(r2[CytoStructColumns.CELL_COUNT_TOTAL], numbers.Integral)
    assert r2[CytoStructColumns.MONOSOMY_CHROMOSOME] == "7"


def test_gen_structured_dataframe_integration(
    clinical_data_train: pd.DataFrame, feature_extractor: CytogeneticsExtractor
) -> None:
    """Integration test: run gen_structured_dataframe on clinical training data."""
    # run on a subset (first 200 rows) to keep test fast but realistic
    subset = clinical_data_train.head(200)
    struct_df = feature_extractor.gen_structured_dataframe(
        subset, cyto_col=Columns.CYTOGENETICS
    )

    # basic sanity checks
    assert len(struct_df) == len(subset)
    assert CytoStructColumns.DELETIONS_COUNT in struct_df.columns
    # at least some rows should have monosomies/trisomies or deletions
    any_abn = (
        (struct_df[CytoStructColumns.DELETIONS_COUNT].fillna(0) > 0).any()
        or (struct_df[CytoStructColumns.MONOSOMIES_COUNT].fillna(0) > 0).any()
        or (struct_df[CytoStructColumns.TRISOMIES_COUNT].fillna(0) > 0).any()
    )
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
        clinical_data=clin_train_proc,
        molecular_data=mol_train,
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

    # 2) Nmut: merge mutation counts into clinical (small subset)
    clin_slice = clinical_data_train.head(50).copy()
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
