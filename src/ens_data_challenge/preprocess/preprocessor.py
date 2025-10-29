import pandas as pd
from typing import List, Union, Any, Optional, Literal, Tuple
import numpy as np

from ens_data_challenge.types import Columns, CytoColumns, MolecularColumns
from .cytogenetic_parser.features import CytogeneticsFeatureExtractor

class Preprocessor:

    def __init__(
            self, 
            clinical_data_train: pd.DataFrame, 
            clinical_data_test: pd.DataFrame,
            molecular_data_train: pd.DataFrame,
            molecular_data_test: pd.DataFrame,
            ):

        self.clinical_data_train = clinical_data_train
        self.clinical_data_test = clinical_data_test

        self.molecular_data_train = molecular_data_train
        self.molecular_data_test = molecular_data_test

    def get_clinical_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Returns the cleaned clinical data."""
        return self.clinical_data_train, self.clinical_data_test

    def get_molecular_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Returns the cleaned molecular data."""
        return self.molecular_data_train, self.molecular_data_test
    
    def clean_data(self) -> None:
        """Cleans the data by handling missing values and removing duplicates."""
        # 1) Extract cytogenetics features and append to clinical tables first
        try:
            self.add_cytogenetics_features()
        except Exception:
            # if cytogenetics feature extraction fails, fail early so the problem is visible
            raise

        # 3) Impute missing values using configured strategies
        self.impute()

        self.clinical_data_train.loc[self.clinical_data_train[CytoColumns.PLOIDY.value] > 50, CytoColumns.PLOIDY.value] = 50
        self.clinical_data_train.loc[self.clinical_data_train[CytoColumns.PLOIDY.value] < 30, CytoColumns.PLOIDY.value] = 30
        self.clinical_data_test.loc[self.clinical_data_test[CytoColumns.PLOIDY.value] > 50, CytoColumns.PLOIDY.value] = 50
        self.clinical_data_test.loc[self.clinical_data_test[CytoColumns.PLOIDY.value] < 30, CytoColumns.PLOIDY.value] = 30

        self._ensure_cat_molecular_consistency(
            categorical_cols = [
                MolecularColumns.GENE.value,
                MolecularColumns.EFFECT.value,
                ],
        )
        
        # 2) Drop unwanted columns used for debugging / irrelevant to model
        self.drop_clinical_columns(columns= [
                Columns.CYTOGENETICS.value,
                Columns.CENTER.value, 
                Columns.MONOCYTES.value, 
                CytoColumns.HAS_LARGE_DELETION.value,
                CytoColumns.HAS_MONOSOMY_Y.value,
                CytoColumns.MDS_IPSS_R_CYTO_RISK.value,
                ])
        
    def drop_molecular_columns(
            self,
            columns: List[str]
            ) -> None:
        """Drops specified columns from both training and testing molecular datasets."""

        self.molecular_data_train.drop(columns=columns, inplace=True)
        self.molecular_data_test.drop(columns=columns, inplace=True)


    def drop_clinical_columns(
            self, 
            columns: List[str]
            ) -> None:
        
        """Drops specified columns from both training and testing datasets."""

        self.clinical_data_train.drop(columns=columns, inplace=True)
        self.clinical_data_test.drop(columns=columns, inplace=True)
            
    def _ensure_cat_clinical_consistency(self, categorical_cols: List[str]) -> None:
        """Ensures that categorical columns have consistent categories between train and test by replacing non-shared categories with 'OTHER'."""
        for col in categorical_cols:
            if col not in self.clinical_data_train.columns or col not in self.clinical_data_test.columns:
                continue

            # Get unique categories in train and test (excluding NaN)
            train_cats = set(self.clinical_data_train[col].dropna().unique())
            test_cats = set(self.clinical_data_test[col].dropna().unique())

            # Categories in train but not in test
            train_only = train_cats - test_cats
            # Categories in test but not in train
            test_only = test_cats - train_cats

            # Replace train_only with 'OTHER' in train
            if train_only:
                self.clinical_data_train.loc[self.clinical_data_train[col].isin(train_only), col] = "OTHER"

            # Replace test_only with 'OTHER' in test
            if test_only:
                self.clinical_data_test.loc[self.clinical_data_test[col].isin(test_only), col] = "OTHER"
    
    def _ensure_cat_molecular_consistency(self, categorical_cols: List[str]) -> None:

        for col in categorical_cols:
            if col not in self.molecular_data_train.columns or col not in self.molecular_data_test.columns:
                continue

            # Get unique categories in train and test (excluding NaN)
            train_cats = set(self.molecular_data_train[col].dropna().unique())
            test_cats = set(self.molecular_data_test[col].dropna().unique())

            # Categories in train but not in test
            train_only = train_cats - test_cats
            # Categories in test but not in train
            test_only = test_cats - train_cats

            # Replace train_only with 'OTHER' in train
            if train_only:
                self.molecular_data_train.loc[self.molecular_data_train[col].isin(train_only), col] = "OTHER"

            # Replace test_only with 'OTHER' in test
            if test_only:
                self.molecular_data_test.loc[self.molecular_data_test[col].isin(test_only), col] = "OTHER"

    def _prune_cat_clinical_outliers(self, categorical_cols: List[str], thresh: float) -> None:
        """Groups infrequent categories in categorical columns by replacing them with 'OTHER'."""
        for col in categorical_cols:
            if col not in self.clinical_data_train.columns:
                continue

            # Frequency pruning on train: replace values with freq <= thresh with 'OTHER'
            freqs = self.clinical_data_train[col].value_counts(normalize=True)
            value_freq = self.clinical_data_train[col].map(freqs).fillna(0.0)
            mask_low_freq_train = value_freq <= thresh
            if mask_low_freq_train.any():
                self.clinical_data_train.loc[mask_low_freq_train, col] = "OTHER"

            # For test: ensure only categories present in train (after pruning) are kept, others to 'OTHER'
            allowed_cats = set(self.clinical_data_train[col].unique())
            test_mask = ~self.clinical_data_test[col].isin(allowed_cats) & self.clinical_data_test[col].notna()
            if test_mask.any():
                self.clinical_data_test.loc[test_mask, col] = "OTHER"

    def _prune_cat_molecular_outliers(self, categorical_cols: List[str], thresh: float) -> None:
        """Groups infrequent categories in categorical columns by replacing them with 'OTHER'."""
        for col in categorical_cols:
            if col not in self.molecular_data_train.columns:
                continue

            # Frequency pruning on train: replace values with freq <= thresh with 'OTHER'
            freqs = self.molecular_data_train[col].value_counts(normalize=True)
            value_freq = self.molecular_data_train[col].map(freqs).fillna(0.0)
            mask_low_freq_train = value_freq <= thresh
            if mask_low_freq_train.any():
                self.molecular_data_train.loc[mask_low_freq_train, col] = "OTHER"

            # For test: ensure only categories present in train (after pruning) are kept, others to 'OTHER'
            allowed_cats = set(self.molecular_data_train[col].unique())
            test_mask = ~self.molecular_data_test[col].isin(allowed_cats) & self.molecular_data_test[col].notna()
            if test_mask.any():
                self.molecular_data_test.loc[test_mask, col] = "OTHER"

    def add_cytogenetics_features(self) -> List[str]:
        """Extracts and adds cytogenetics features from the CYTOGENETICS column."""

        feature_extractor = CytogeneticsFeatureExtractor()

        # Add features directly to clinical_data_train / clinical_data_test (they must contain CYTOGENETICS)
        cyto_feat_train, new_cols_train = feature_extractor.gen_features_to_dataframe(
            self.clinical_data_train, cyto_col='CYTOGENETICS'
        )
        cyto_feat_test, new_cols_test = feature_extractor.gen_features_to_dataframe(
            self.clinical_data_test, cyto_col='CYTOGENETICS'
        )

        if set(new_cols_train) != set(new_cols_test):
            raise ValueError("Les features cytogénétiques créées pour l'ensemble d'entraînement et de test ne correspondent pas.")
        
        self.clinical_data_train = self.clinical_data_train.drop(columns=new_cols_train, errors='ignore')
        self.clinical_data_test = self.clinical_data_test.drop(columns=new_cols_test, errors='ignore')

        self.clinical_data_train = pd.concat([self.clinical_data_train, cyto_feat_train], axis=1)
        self.clinical_data_test = pd.concat([self.clinical_data_test, cyto_feat_test], axis=1)

        return new_cols_train
    
    def impute(self) -> None:
        """Imputes missing values in the CYTOGENETICS column specifically for TP53."""

        self._impute_molecular_missing_values(
            cols=[
                MolecularColumns.GENE.value, 
                MolecularColumns.REF.value, 
                MolecularColumns.ALT.value, 
                MolecularColumns.CHR.value
            ],
            strategy="UNKNOWN"
        )

        self._impute_molecular_missing_values(
            cols=[MolecularColumns.EFFECT.value],
            strategy="most_frequent"
        )

        self._impute_molecular_missing_values(
            cols=[MolecularColumns.VARIANT_ALLELE_FREQUENCY.value, MolecularColumns.DEPTH.value],
            strategy="median"
        )

        # Impute missing values for cytogenetics-related categorical columns
        self._impute_clinical_missing_values(
            cols= [
                CytoColumns.HAS_MONOSOMY_7.value,
                CytoColumns.IS_NORMAL.value,
                CytoColumns.PLOIDY.value,
                CytoColumns.HAS_TP53_DELETION.value,
                CytoColumns.HAS_COMPLEX_CHR3.value,
                CytoColumns.N_ABNORMALITIES.value,
                CytoColumns.N_CHROMOSOMES_AFFECTED.value,
                CytoColumns.HAS_DEL_5Q.value,
                CytoColumns.HAS_DEL_7Q.value,
                CytoColumns.N_DELETIONS.value,
                CytoColumns.N_CRITICAL_REGIONS_DELETED.value,
                CytoColumns.IS_MOSAIC.value,
                CytoColumns.N_CLONES.value,
                CytoColumns.ABNORMAL_CLONE_PERCENTAGE.value,
                CytoColumns.MDS_IPSS_CYTO_RISK.value,
                CytoColumns.AML_ELN_2022_CYTO_RISK.value,
                CytoColumns.CLL_CYTO_RISK.value,
                CytoColumns.MM_RISS_CYTO_RISK.value,
                ],
            strategy="most_frequent"
            )
        
        # Impute missing values for numerical columns
        self._impute_clinical_missing_values(
            cols=[
                CytoColumns.COMPUTED_RISK_SCORE.value,
                Columns.BM_BLAST.value,
                Columns.WBC.value,
                Columns.ANC.value,
                Columns.HB.value,
                Columns.PLT.value,
                ],
            strategy="median"
            )
    
    def _impute_molecular_missing_values(
            self, 
            cols: List[str], 
            strategy: Literal["median", "mean", "most_frequent", "UNKNOWN"]
            ) -> None:
        """Imputes missing values in both training and testing datasets using the specified strategy."""

        if strategy == "mean":
            for col in cols:
                mean_val = self.molecular_data_train[col].mean()
                self.molecular_data_train.fillna({col: mean_val}, inplace=True)
                self.molecular_data_test.fillna({col: mean_val}, inplace=True)
        
        elif strategy == "median":
            for col in cols:
                median_val = self.molecular_data_train[col].median()
                self.molecular_data_train.fillna({col: median_val}, inplace=True)
                self.molecular_data_test.fillna({col: median_val}, inplace=True)

        elif strategy == "most_frequent":
            for col in cols:
                mode_val = self.molecular_data_train[col].mode()[0]
                self.molecular_data_train.fillna({col: mode_val}, inplace=True)
                self.molecular_data_test.fillna({col: mode_val}, inplace=True)
        
        elif strategy == "UNKNOWN":
            for col in cols:
                self.molecular_data_train.fillna({col: "UNKNOWN"}, inplace=True)
                self.molecular_data_test.fillna({col: "UNKNOWN"}, inplace=True)

    def _impute_clinical_missing_values(
            self, 
            cols: List[str], 
            strategy: Literal["median", "mean", "most_frequent", "UNKNOWN"]
            ) -> None:
        """Imputes missing values in both training and testing datasets using the specified strategy."""

        if strategy == "mean":
            for col in cols:
                mean_val = self.clinical_data_train[col].mean()
                self.clinical_data_train.fillna({col: mean_val}, inplace=True)
                self.clinical_data_test.fillna({col: mean_val}, inplace=True)
        
        elif strategy == "median":
            for col in cols:
                median_val = self.clinical_data_train[col].median()
                self.clinical_data_train.fillna({col: median_val}, inplace=True)
                self.clinical_data_test.fillna({col: median_val}, inplace=True)
        
        elif strategy == "most_frequent":
            for col in cols:
                mode_val = self.clinical_data_train[col].mode()[0]
                self.clinical_data_train.fillna({col: mode_val}, inplace=True)
                self.clinical_data_test.fillna({col: mode_val}, inplace=True)
        
        elif strategy == "UNKNOWN":
            for col in cols:
                self.clinical_data_train.fillna({col: "UNKNOWN"}, inplace=True)
                self.clinical_data_test.fillna({col: "UNKNOWN"}, inplace=True)