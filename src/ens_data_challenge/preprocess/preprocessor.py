from typing import Any, Dict, List, Literal, Optional, Tuple, Union
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import re

from .cytogenetic_parser.dataframe import CytogeneticsExtractor
from ens_data_challenge.types import Columns, CytoColumns, MolecularColumns, CytoStructColumns

class Preprocessor(BaseEstimator, TransformerMixin):

    def __init__(self, cat_thresh: float = 0.01):

        self.cat_thresh = cat_thresh
        # stocker les valeurs apprises pendant fit
        self.molecular_impute_values: Dict[str, Any] = {}
        self.clinical_impute_values: Dict[str, Any] = {}
        self.cyto_struct_impute_values: Dict[str, Any] = {}
        self.allowed_cat_molecular: Dict[str, List[str]] = {}
        self.allowed_cat_clinical: Dict[str, List[str]] = {}
        self.allowed_cat_cyto_struct: Dict[str, List[str]] = {}
    
    def fit(
        self,
        clinical_data_train: pd.DataFrame,
        molecular_data_train: pd.DataFrame,
        clinical_data_test: pd.DataFrame,
        molecular_data_test: pd.DataFrame
    ):
        # Extract cytogenetics features for train
        feature_extractor = CytogeneticsExtractor()

        cyto_feat_train, _ = feature_extractor.gen_features_to_dataframe(
            clinical_data_train, cyto_col='CYTOGENETICS'
        )

        cyto_feat_test, _ = feature_extractor.gen_features_to_dataframe(
            clinical_data_test, cyto_col='CYTOGENETICS'
        )
        
        cyto_struct_train, _ = feature_extractor.gen_structured_dataframe(
            clinical_data_train, cyto_col='CYTOGENETICS'
        )

        cyto_struct_test, _ = feature_extractor.gen_structured_dataframe(
            clinical_data_test, cyto_col='CYTOGENETICS'
        )

        clinical_data_train_with_cyto = pd.concat([clinical_data_train, cyto_feat_train], axis=1)
        clinical_data_test_with_cyto = pd.concat([clinical_data_test, cyto_feat_test], axis=1)

        # --- imputation ---
        self._fit_imputation(
            clinical_data_train_with_cyto,
            molecular_data_train,
            cyto_struct_train
            )

        # --- categorical consistency ---
        self._ensure_categorical_consistency(
            molecular_data_train,
            molecular_data_test,
            cols=[
                MolecularColumns.GENE.value, 
                MolecularColumns.EFFECT.value,
                MolecularColumns.REF.value,
                MolecularColumns.ALT.value,
                MolecularColumns.CHR.value
            ],
            data_type="molecular"
        )

        self._ensure_categorical_consistency(
            clinical_data_train_with_cyto,
            clinical_data_test_with_cyto,
            cols=[
                CytoColumns.IS_NORMAL.value,
                CytoColumns.HAS_TP53_DELETION.value,
                CytoColumns.HAS_COMPLEX_CHR3.value
            ],
            data_type="clinical"
        )

        self._ensure_categorical_consistency(
            cyto_struct_train,
            cyto_struct_test,
            cols=[
                CytoStructColumns.PLOIDY.value,
            ],
            data_type="cyto_struct"
        )

        return cyto_struct_train, cyto_struct_test

    def transform(
        self, 
        clinical_data: pd.DataFrame, 
        molecular_data: pd.DataFrame,
        cyto_struct: pd.DataFrame,
        targets: Optional[pd.DataFrame] = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
        
        clinical_data = clinical_data.copy()
        molecular_data = molecular_data.copy()
        cyto_struct = cyto_struct.copy()

        # Extract cytogenetics features for train
        feature_extractor = CytogeneticsExtractor()

        cyto_feat, _ = feature_extractor.gen_features_to_dataframe(
            clinical_data, cyto_col='CYTOGENETICS'
        )

        cyto_struct, _ = feature_extractor.gen_structured_dataframe(
            clinical_data, cyto_col='CYTOGENETICS'
        )

        clinical_data_with_cyto = pd.concat([clinical_data, cyto_feat], axis=1)

        # imputation
        self._apply_imputation(molecular_data, clinical_data_with_cyto, cyto_struct)

        # categorical consistency
        self._apply_categorical_consistency(molecular_data, clinical_data_with_cyto, cyto_struct)

        drop_clinical_cols = [
            Columns.CYTOGENETICS.value, 
            Columns.MONOCYTES.value,
            ]
        clinical_data_with_cyto.drop(columns=drop_clinical_cols, errors='ignore', inplace=True)

        if targets is not None:
                
            # 2. Vérifier le masque
            mask = targets.isna().any(axis=1)

            # 3. Vérifier targets APRÈS nettoyage
            targets_clean = targets[~mask].reset_index(drop=True)
            
            remain_ids = targets_clean['ID'].values

            clinical_data_with_cyto = clinical_data_with_cyto[clinical_data_with_cyto['ID'].isin(remain_ids)].reset_index(drop=True)
            molecular_data = molecular_data[molecular_data['ID'].isin(remain_ids)].reset_index(drop=True)
            cyto_struct = cyto_struct[cyto_struct['ID'].isin(remain_ids)].reset_index(drop=True)

            return clinical_data_with_cyto, molecular_data, cyto_struct, targets_clean
        
        return clinical_data_with_cyto, molecular_data, cyto_struct

    # --------------------------
    # internal helper functions
    # --------------------------
    def _fit_imputation(self, clinical_data: pd.DataFrame, molecular_data: pd.DataFrame, cyto_struct: pd.DataFrame):
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
            (MolecularColumns.END.value, "zero")
        ]:
            self.molecular_impute_values[col] = self._compute_impute_value(
                molecular_data[col], strategy
            )

        # Clinical
        self.clinical_impute_values = {}
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
            CytoColumns.MDS_IPSS_R_CYTO_RISK.value,
            CytoColumns.AML_ELN_2022_CYTO_RISK.value,
            CytoColumns.CLL_CYTO_RISK.value,
            CytoColumns.MM_RISS_CYTO_RISK.value
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
            Columns.PLT.value
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
            CytoStructColumns.CLONE_CELL_COUNT.value
        ]:
            self.cyto_struct_impute_values[col] = self._compute_impute_value(
                cyto_struct[col], "zero"
            )

    def _compute_impute_value(self, series: pd.Series, strategy: Literal["median", "most_frequent", "UNKNOWN", "zero"]):
        if strategy == "median":
            return series.median()
        elif strategy == "most_frequent":
            return series.mode()[0]
        elif strategy == "UNKNOWN":
            return "UNKNOWN"
        elif strategy == "zero":
            return 0
        else:
            raise ValueError(strategy)

    def _apply_imputation(self, molecular_data: pd.DataFrame, clinical_data: pd.DataFrame, cyto_struct: pd.DataFrame):
        """Applique l'imputation des valeurs manquantes aux données moléculaires et cliniques."""
        self._impute_dataframe(molecular_data, self.molecular_impute_values)
        self._impute_dataframe(clinical_data, self.clinical_impute_values)
        self._impute_dataframe(cyto_struct, self.cyto_struct_impute_values)

    def _impute_dataframe(self, df: pd.DataFrame, impute_values: dict):
        """Impute les valeurs manquantes d'un DataFrame selon un dictionnaire de valeurs."""
        for col, val in impute_values.items():
            mask = df[col].isna()
            df.loc[mask, col] = val

    def _ensure_categorical_consistency(
        self,
        test_data: pd.DataFrame,
        train_data: pd.DataFrame,
        cols: List[str],
        data_type: Literal["molecular", "clinical", "cyto_struct"]
    ):
        
        # Molecular
        for col in cols:
            allowed_train = set(train_data[col].dropna().unique())
            allowed_test = set(test_data[col].dropna().unique())
            allowed = allowed_train.intersection(allowed_test)

            if data_type == "molecular":
                self.allowed_cat_molecular[col] = list(allowed)
            elif data_type == "clinical":
                self.allowed_cat_clinical[col] = list(allowed)
            elif data_type == "cyto_struct":
                self.allowed_cat_cyto_struct[col] = list(allowed)

    def _apply_categorical_consistency(
            self, 
            molecular: pd.DataFrame, 
            clinical: pd.DataFrame, 
            cyto_struct: pd.DataFrame
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