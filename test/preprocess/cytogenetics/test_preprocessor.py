import pytest
import pandas as pd
import numpy as np
from ens_data_challenge.preprocess import Preprocessor
from ens_data_challenge.globals import (
    TRAIN_CLINICAL_DATA_PATH,
    TEST_CLINICAL_DATA_PATH,
    TRAIN_MOLECULAR_DATA_PATH,
    TEST_MOLECULAR_DATA_PATH,
    TRAIN_TARGET_PATH,
)
from ens_data_challenge.types import (
    Columns,
    MolecularColumns,
    CytoColumns,
)


@pytest.fixture
def sample_data() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    # Load real data but limit to small subset for testing
    clinical_train = pd.read_csv(TRAIN_CLINICAL_DATA_PATH).head(50).copy()
    clinical_test = pd.read_csv(TEST_CLINICAL_DATA_PATH).head(20).copy()
    molecular_train = pd.read_csv(TRAIN_MOLECULAR_DATA_PATH).head(100).copy()
    molecular_test = pd.read_csv(TEST_MOLECULAR_DATA_PATH).head(50).copy()
    targets_train = pd.read_csv(TRAIN_TARGET_PATH).head(50).copy()

    # Filter molecular data to only include IDs present in clinical data subset for consistency in tests
    train_ids = clinical_train["ID"].unique()
    test_ids = clinical_test["ID"].unique()

    molecular_train = molecular_train[molecular_train["ID"].isin(train_ids)].copy()
    molecular_test = molecular_test[molecular_test["ID"].isin(test_ids)].copy()
    targets_train = targets_train[targets_train["ID"].isin(train_ids)].copy()

    # For cytogenetics structured data, we can create empty dataframes with IDs for testing
    prep = Preprocessor()
    clinical_train, cyto_struct_train = prep.get_cyto_features_and_df(clinical_train)
    clinical_test, cyto_struct_test = prep.get_cyto_features_and_df(clinical_test)

    return (
        clinical_train,
        molecular_train,
        targets_train,
        clinical_test,
        molecular_test,
        cyto_struct_train,
        cyto_struct_test,
    )


def test_preprocessor_fit_transform(sample_data):
    (
        clinical_train,
        molecular_train,
        targets_train,
        clinical_test,
        molecular_test,
        cyto_struct_train,
        cyto_struct_test,
    ) = sample_data

    preprocessor = Preprocessor()

    # FIT
    (
        clinical_train,
        molecular_train,
        cyto_struct_train,
        clinical_test,
        molecular_test,
        cyto_struct_test,
        targets_train,
    ) = preprocessor.fit_transform(
        clinical_train,
        molecular_train,
        clinical_test,
        molecular_test,
        cyto_struct_train,
        cyto_struct_test,
        targets_train,
    )

    # Check attributes populated
    assert hasattr(preprocessor, "molecular_impute_values")
    assert (
        MolecularColumns.VARIANT_ALLELE_FREQUENCY.value
        in preprocessor.molecular_impute_values
    )
    assert hasattr(preprocessor, "clinical_impute_values")
    assert Columns.BM_BLAST.value in preprocessor.clinical_impute_values

    # TRANSFORM
    clin_res, mol_res, cyto_res = preprocessor.transform(
        clinical_train, molecular_train, cyto_struct_train
    )

    # 1. Check Imputation
    assert not clin_res[Columns.BM_BLAST.value].isnull().any()
    assert not mol_res[MolecularColumns.VARIANT_ALLELE_FREQUENCY.value].isnull().any()

    # 2. Check Feature Engineering
    assert CytoColumns.N_ABNORMALITIES.value in clin_res.columns

    # 3. Check targets consistency
    # All IDs present in targets should be kept if targets has no NaNs (assuming clean input or handled)
    # Note: Real data might have NaNs in input params but our preprocessor handles them.
    # We just ensure output rows match.
    assert len(targets_train) == len(clin_res)


def test_target_filtering(sample_data):
    (
        clinical_train,
        molecular_train,
        targets_train,
        clinical_test,
        molecular_test,
        cyto_struct_train,
        cyto_struct_test,
    ) = sample_data

    # Make one target NaN if possible
    if len(targets_train) > 0:
        target_id_to_nan = targets_train.iloc[0]["ID"]
        targets_train.loc[targets_train["ID"] == target_id_to_nan, "y"] = np.nan

        preprocessor = Preprocessor()
        preprocessor.fit(
            clinical_train,
            molecular_train,
            clinical_test,
            molecular_test,
            cyto_struct_train,
            cyto_struct_test,
            targets_train,
        )

        clin_res, mol_res, cyto_res = preprocessor.transform(
            clinical_train, molecular_train, cyto_struct_train
        )

        # That ID should be dropped
        assert target_id_to_nan not in preprocessor.remains_ids
        assert target_id_to_nan not in clin_res[Columns.ID.value].values


def test_categorical_consistency(sample_data):
    (
        clinical_train,
        molecular_train,
        targets_train,
        clinical_test,
        molecular_test,
        cyto_struct_train,
        cyto_struct_test,
    ) = sample_data

    # Modifying real dataframes for the test
    # Ensure they have enough rows
    if len(molecular_train) < 2 or len(molecular_test) < 2:
        pytest.skip("Not enough data to run categorical consistency test")

    molecular_train = molecular_train.copy()
    molecular_test = molecular_test.copy()

    # Inject categories
    # Train has 'TRAIN_ONLY'
    # Test has 'TEST_ONLY'
    # Both have 'COMMON'
    # We overwrite the GENE column for the first few rows

    mol_train_idx0 = molecular_train.index[0]
    mol_train_idx1 = molecular_train.index[1]

    molecular_train.at[mol_train_idx0, MolecularColumns.GENE.value] = "TRAIN_ONLY"
    molecular_train.at[mol_train_idx1, MolecularColumns.GENE.value] = "COMMON"

    mol_test_idx0 = molecular_test.index[0]
    mol_test_idx1 = molecular_test.index[1]

    molecular_test.at[mol_test_idx0, MolecularColumns.GENE.value] = "TEST_ONLY"
    molecular_test.at[mol_test_idx1, MolecularColumns.GENE.value] = "COMMON"

    preprocessor = Preprocessor()
    preprocessor.fit(
        clinical_train,
        molecular_train,
        clinical_test,
        molecular_test,
        cyto_struct_train,
        cyto_struct_test,
        targets_train,
    )

    # Check allowed categories
    allowed_genes = preprocessor.allowed_cat_molecular[MolecularColumns.GENE.value]
    assert "COMMON" in allowed_genes
    assert "TRAIN_ONLY" not in allowed_genes
    assert "TEST_ONLY" not in allowed_genes

    # Transform Train
    # We need a targets dataframe that matches the clinical IDs
    # Since we are just testing categorical consistency on molecular data, we can use the original targets
    # provided they match ID.

    _, mol_res_train, _ = preprocessor.transform(
        clinical_train, molecular_train, cyto_struct_train
    )

    # 'TRAIN_ONLY' should become 'OTHER' because it wasn't in Test
    _genes_res = mol_res_train[MolecularColumns.GENE.value]

    # Need to check the specific row where we injected 'TRAIN_ONLY'
    # Find the ID for that row
    id_train_only = molecular_train.at[mol_train_idx0, "ID"]

    # If that ID remains in the dataset (has valid target), check it
    if id_train_only in mol_res_train["ID"].values:
        gene_val = mol_res_train.loc[
            mol_res_train["ID"] == id_train_only, MolecularColumns.GENE.value
        ]
        # It might be multiple rows for that ID, check if 'OTHER' is present
        assert "OTHER" in gene_val.values or "COMMON" in gene_val.values  # type: ignore
        assert "TRAIN_ONLY" not in gene_val.values  # type: ignore
