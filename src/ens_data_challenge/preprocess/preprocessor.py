from typing import Any, Dict, List, Literal, Tuple
import pandas as pd
from sklearn.base import BaseEstimator

from .cytogenetic_parser.dataframe import CytogeneticsExtractor
from ens_data_challenge.types import (
    Columns,
    CytoColumns,
    MolecularColumns,
    CytoStructColumns,
    MdsIpssRCytoRisk,
    AmlEln2022CytoRisk,
    CllCytoRisk,
    MmRissCytoRisk,
)


class Preprocessor(BaseEstimator):
    def __init__(self, cat_thresh: float = 0.01):
        self.cat_thresh = cat_thresh
        # stocker les valeurs apprises pendant fit
        self.molecular_impute_values: Dict[str, Any] = {}
        self.clinical_impute_values: Dict[str, Any] = {}
        self.cyto_struct_impute_values: Dict[str, Any] = {}
        self.allowed_cat_molecular: Dict[str, List[str]] = {}
        self.allowed_cat_clinical: Dict[str, List[str]] = {}
        self.allowed_cat_cyto_struct: Dict[str, List[str]] = {}
        self.remains_ids: List[Any] = []

    def get_cyto_features_and_df(
        self, clinical_data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate cytogenetics features from clinical data.
        Args:
            clinical_data (pd.DataFrame): Clinical data containing cytogenetics information.
        Returns:
                pd.DataFrame: Clinical data with added cytogenetics features.
                pd.DataFrame: Structured cytogenetics data with cytogenetics information.
        """
        clinical_data = clinical_data.copy()

        feature_extractor = CytogeneticsExtractor()

        cyto_feat = feature_extractor.gen_features_to_dataframe(
            clinical_data, cyto_col="CYTOGENETICS"
        )

        cyto_struct = feature_extractor.gen_structured_dataframe(
            clinical_data, cyto_col="CYTOGENETICS"
        )

        clinical_data_with_cyto = pd.concat(
            [clinical_data.drop(columns=cyto_feat.columns, errors="ignore"), cyto_feat],
            axis=1,
        )

        return clinical_data_with_cyto, cyto_struct

    def fit(
        self,
        clinical_data_train: pd.DataFrame,
        molecular_data_train: pd.DataFrame,
        clinical_data_test: pd.DataFrame,
        molecular_data_test: pd.DataFrame,
        cyto_struct_train: pd.DataFrame,
        cyto_struct_test: pd.DataFrame,
        targets: pd.DataFrame,
    ) -> None:
        """Fit the preprocessor on training data and ensure consistency with test data.
        Needs to be called before transform."""

        # --- imputation ---
        self._fit_imputation(
            clinical_data_train, molecular_data_train, cyto_struct_train
        )

        # --- categorical consistency ---
        self._fit_categorical_consistency(
            molecular_data_train,
            molecular_data_test,
            cols=[
                MolecularColumns.GENE.value,
                MolecularColumns.EFFECT.value,
                MolecularColumns.REF.value,
                MolecularColumns.ALT.value,
                MolecularColumns.CHR.value,
            ],
            data_type="molecular",
        )

        self._fit_categorical_consistency(
            clinical_data_train,
            clinical_data_test,
            cols=[
                CytoColumns.IS_NORMAL.value,
                CytoColumns.HAS_TP53_DELETION.value,
                CytoColumns.HAS_COMPLEX_CHR3.value,
            ],
            data_type="clinical",
        )

        self._fit_categorical_consistency(
            cyto_struct_train,
            cyto_struct_test,
            cols=[
                CytoStructColumns.PLOIDY.value,
            ],
            data_type="cyto_struct",
        )

        # Nettoyer les targets
        targets = targets.copy()

        # 2. Vérifier le masque
        mask = targets.isna().any(axis=1)

        # 3. Vérifier targets APRÈS nettoyage
        targets_clean = targets[~mask].reset_index(drop=True)

        self.remains_ids = targets_clean["ID"].values.tolist()

        return

    def clean_targets(
        self,
        targets: pd.DataFrame,
    ) -> pd.DataFrame:
        """Clean the targets by removing rows with missing values.
        Args:
            targets (pd.DataFrame): Target labels.
        Returns:
            pd.DataFrame: Cleaned targets.
        """

        targets = targets.copy()

        # 2. Vérifier le masque
        mask = targets.isna().any(axis=1)

        # 3. Vérifier targets APRÈS nettoyage
        targets_clean = targets[~mask].reset_index(drop=True)

        return targets_clean

    def fit_transform(
        self,
        clinical_data: pd.DataFrame,
        molecular_data: pd.DataFrame,
        clinical_data_test: pd.DataFrame,
        molecular_data_test: pd.DataFrame,
        cyto_struct_train: pd.DataFrame,
        cyto_struct_test: pd.DataFrame,
        targets: pd.DataFrame,
    ) -> Tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
    ]:
        """Fit the preprocessor and transform the data in one step.
        Args:
            clinical_data (pd.DataFrame): Clinical data.
            molecular_data (pd.DataFrame): Molecular data.
            clinical_data_test (pd.DataFrame): Clinical test data.
            molecular_data_test (pd.DataFrame): Molecular test data.
            cyto_struct_data (pd.DataFrame): Cytogenetics structured data.
            targets (pd.DataFrame): Target labels.
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
                - Cleaned clinical training data with cytogenetics features.
                - Cleaned molecular training data.
                - Cleaned cytogenetics structured training data.
                - Cleaned clinical test data with cytogenetics features.
                - Cleaned molecular test data.
                - Cleaned targets."""

        self.fit(
            clinical_data,
            molecular_data,
            clinical_data_test,
            molecular_data_test,
            cyto_struct_train,
            cyto_struct_test,
            targets,
        )

        targets = self.clean_targets(targets)

        clinical_data, molecular_data, cyto_struct_train = self.transform(
            clinical_data, molecular_data, cyto_struct_train
        )

        clinical_data_test, molecular_data_test, cyto_struct_test = self.transform(
            clinical_data_test, molecular_data_test, cyto_struct_test
        )

        return (
            clinical_data,
            molecular_data,
            clinical_data_test,
            molecular_data_test,
            cyto_struct_train,
            cyto_struct_test,
            targets,
        )

    def transform(
        self,
        clinical_data_with_cyto: pd.DataFrame,
        molecular_data: pd.DataFrame,
        cyto_struct: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Transform the data by applying imputation and categorical consistency.
        Also cleans the targets by removing rows with missing values. get_cyto_features_and_df
        should be called before this method to get the clinical data with cytogenetics features.

        Args:
            clinical_data_with_cyto (pd.DataFrame): Clinical data with cytogenetics features.
            molecular_data (pd.DataFrame): Molecular data.
            cyto_struct (pd.DataFrame): Cytogenetics structured data.
            targets (pd.DataFrame): Target labels.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
                - Cleaned clinical data with cytogenetics features.
                - Cleaned molecular data.
                - Cleaned cytogenetics structured data.
        """

        clinical_data = clinical_data_with_cyto.copy()
        molecular_data = molecular_data.copy()
        cyto_struct = cyto_struct.copy()

        # imputation
        self._apply_imputation(molecular_data, clinical_data, cyto_struct)

        # categorical consistency
        self._apply_categorical_consistency(molecular_data, clinical_data, cyto_struct)

        drop_clinical_cols = [
            Columns.CYTOGENETICS.value,
            Columns.MONOCYTES.value,
        ]
        clinical_data.drop(columns=drop_clinical_cols, errors="ignore", inplace=True)

        clinical_data = clinical_data[
            clinical_data["ID"].isin(self.remains_ids)
        ].reset_index(drop=True)
        molecular_data = molecular_data[
            molecular_data["ID"].isin(self.remains_ids)
        ].reset_index(drop=True)
        cyto_struct = cyto_struct[cyto_struct["ID"].isin(self.remains_ids)].reset_index(
            drop=True
        )

        return clinical_data, molecular_data, cyto_struct

    # --------------------------
    # internal helper functions
    # --------------------------
    def _fit_imputation(
        self,
        clinical_data: pd.DataFrame,
        molecular_data: pd.DataFrame,
        cyto_struct: pd.DataFrame,
    ):
        """Calcule les valeurs d'imputation pour les données moléculaires et cliniques."""

        # Molecular
        self.molecular_impute_values = {}
        for col, strategy in [
            (MolecularColumns.GENE.value, "UNKNOWN"),
            (MolecularColumns.REF.value, "UNKNOWN"),
            (MolecularColumns.ALT.value, "UNKNOWN"),
            (MolecularColumns.CHR.value, "zero"),
            (MolecularColumns.EFFECT.value, "most_frequent"),
            (MolecularColumns.VARIANT_ALLELE_FREQUENCY.value, "median"),
            (MolecularColumns.DEPTH.value, "median"),
            (MolecularColumns.START.value, "zero"),
            (MolecularColumns.END.value, "zero"),
        ]:
            self.molecular_impute_values[col] = self._compute_impute_value(
                molecular_data[col],
                strategy,  # type: ignore
            )

        # Clinical
        self.clinical_impute_values = {}

        # Risk columns with specific Enums
        risk_defaults = {
            CytoColumns.MDS_IPSS_R_CYTO_RISK.value: MdsIpssRCytoRisk.UNKNOWN,
            CytoColumns.AML_ELN_2022_CYTO_RISK.value: AmlEln2022CytoRisk.UNKNOWN,
            CytoColumns.CLL_CYTO_RISK.value: CllCytoRisk.UNKNOWN,
            CytoColumns.MM_RISS_CYTO_RISK.value: MmRissCytoRisk.UNKNOWN,
        }

        for col, default_enum in risk_defaults.items():
            self.clinical_impute_values[col] = self._compute_impute_value(
                clinical_data[col], "most_frequent", default_if_empty=default_enum
            )

        for col in [
            CytoColumns.IS_NORMAL.value,
            CytoColumns.HAS_TP53_DELETION.value,
            CytoColumns.HAS_COMPLEX_CHR3.value,
            CytoColumns.N_ABNORMALITIES.value,
            CytoColumns.N_CHROMOSOMES_AFFECTED.value,
            CytoColumns.N_DELETIONS.value,
            CytoColumns.N_CRITICAL_REGIONS_DELETED.value,
            CytoColumns.IS_MOSAIC.value,
            CytoColumns.N_CLONES.value,
            CytoColumns.ABNORMAL_CLONE_PERCENTAGE.value,
        ]:
            self.clinical_impute_values[col] = self._compute_impute_value(
                clinical_data[col], "most_frequent"
            )

        for col in [
            CytoColumns.COMPUTED_RISK_SCORE.value,
            Columns.BM_BLAST.value,
            Columns.WBC.value,
            Columns.ANC.value,
            Columns.HB.value,
            Columns.PLT.value,
        ]:
            self.clinical_impute_values[col] = self._compute_impute_value(
                clinical_data[col], "median"
            )

        for col in [
            CytoStructColumns.SEX_CHROMOSOMES.value,
            CytoStructColumns.ARM.value,
            CytoStructColumns.MUTATION_TYPE.value,
            CytoStructColumns.START_ARM.value,
            CytoStructColumns.END_ARM.value,
        ]:
            self.cyto_struct_impute_values[col] = self._compute_impute_value(
                cyto_struct[col], "UNKNOWN"
            )

        for col in [
            CytoStructColumns.PLOIDY.value,
            CytoStructColumns.START.value,
            CytoStructColumns.END.value,
            CytoStructColumns.CHROMOSOME.value,
            CytoStructColumns.CLONE_INDEX.value,
            CytoStructColumns.CLONE_CELL_COUNT.value,
        ]:
            self.cyto_struct_impute_values[col] = self._compute_impute_value(
                cyto_struct[col], "zero"
            )

    def _compute_impute_value(
        self,
        series: pd.Series,
        strategy: Literal["median", "most_frequent", "UNKNOWN", "zero"],
        default_if_empty: Any = "UNKNOWN",
    ):
        if strategy == "median":
            return series.median()
        elif strategy == "most_frequent":
            modes = series.mode()
            if not modes.empty:
                return modes[0]
            return default_if_empty
        elif strategy == "UNKNOWN":
            return default_if_empty
        elif strategy == "zero":
            return 0
        else:
            raise ValueError(strategy)

    def _apply_imputation(
        self,
        molecular_data: pd.DataFrame,
        clinical_data: pd.DataFrame,
        cyto_struct: pd.DataFrame,
    ):
        """Applique l'imputation des valeurs manquantes aux données moléculaires et cliniques."""
        self._impute_dataframe(molecular_data, self.molecular_impute_values)
        self._impute_dataframe(clinical_data, self.clinical_impute_values)
        self._impute_dataframe(cyto_struct, self.cyto_struct_impute_values)

    def _impute_dataframe(self, df: pd.DataFrame, impute_values: dict):
        """Impute les valeurs manquantes d'un DataFrame selon un dictionnaire de valeurs."""
        for col, val in impute_values.items():
            # Ensure col is string to avoid issues with StrEnum being treated as list-like in loc
            col_str = str(col)
            if col_str in df.columns:
                mask = df[col_str].isna()
                df.loc[mask, col_str] = val

    def _fit_categorical_consistency(
        self,
        test_data: pd.DataFrame,
        train_data: pd.DataFrame,
        cols: List[str],
        data_type: Literal["molecular", "clinical", "cyto_struct"],
    ):
        # Molecular
        for col in cols:
            allowed_train = set(train_data[col].dropna())
            allowed_test = set(test_data[col].dropna())
            allowed = allowed_train.intersection(allowed_test)

            if data_type == "molecular":
                self.allowed_cat_molecular[col] = list(allowed)
            elif data_type == "clinical":
                self.allowed_cat_clinical[col] = list(allowed)
            elif data_type == "cyto_struct":
                self.allowed_cat_cyto_struct[col] = list(allowed)

    def _apply_categorical_consistency(
        self, molecular: pd.DataFrame, clinical: pd.DataFrame, cyto_struct: pd.DataFrame
    ):
        for col, allowed in self.allowed_cat_molecular.items():
            molecular[col] = molecular[col].where(
                molecular[col].isin(allowed), other="OTHER"
            )

        for col, allowed in self.allowed_cat_clinical.items():
            clinical[col] = clinical[col].where(
                clinical[col].isin(allowed), other="OTHER"
            )

        for col, allowed in self.allowed_cat_cyto_struct.items():
            cyto_struct[col] = cyto_struct[col].where(
                cyto_struct[col].isin(allowed), other="OTHER"
            )
