from typing import Any, Dict, List, Optional, Literal, Tuple
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import OneHotEncoder

from ens_data_challenge.types import HighRiskGenes, PATHWAY_GENES
from .utils.genes_coefs_calculator import GeneEnhancedEncoder

class FeatureEngineer2:

    def __init__(
            self, 
            clinical_data_train: pd.DataFrame, 
            molecular_data_train: pd.DataFrame,
            cytogenetic_data_train: pd.DataFrame,
            clinical_data_test: pd.DataFrame,
            molecular_data_test: pd.DataFrame,
            cytogenetic_data_test: pd.DataFrame,
            target_data_train: pd.DataFrame,
            ) :

        self.molecular_train = molecular_data_train.copy()
        self.molecular_test = molecular_data_test.copy()

        self.cytogenetic_train = cytogenetic_data_train.copy()
        self.cytogenetic_test = cytogenetic_data_test.copy()

        self.target_train = target_data_train.copy()


        # Data (copied to avoid modifying input)
        self.clinical_train = clinical_data_train.copy()
        self.clinical_test = clinical_data_test.copy()

    def get_clinical_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train_df = self.clinical_train.select_dtypes(include=[np.number])
        test_df = self.clinical_test.select_dtypes(include=[np.number])
        train_df['ID'] = self.clinical_train['ID']
        test_df['ID'] = self.clinical_test['ID']
        return train_df, test_df

    def get_molecular_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train_df = self.molecular_train.select_dtypes(include=[np.number])
        test_df = self.molecular_test.select_dtypes(include=[np.number])
        train_df['ID'] = self.molecular_train['ID']
        test_df['ID'] = self.molecular_test['ID']
        return train_df, test_df

    def get_cytogenetic_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train_df = self.cytogenetic_train.select_dtypes(include=[np.number])
        test_df = self.cytogenetic_test.select_dtypes(include=[np.number])
        train_df['ID'] = self.cytogenetic_train['ID']
        test_df['ID'] = self.cytogenetic_test['ID']
        return train_df, test_df

    def as_type(self, type_dict: Dict[str, Any], data_type: Literal["molecular", "clinical", "cytogenetic"]) -> List[str]:
        """Convert column to specified data type in the specified dataframes."""
        if data_type == "clinical":
            for col, dtype in type_dict.items():
                self.clinical_train[col] = self.clinical_train[col].astype(dtype)
                self.clinical_test[col] = self.clinical_test[col].astype(dtype)
        elif data_type == "molecular":
            for col, dtype in type_dict.items():
                self.molecular_train[col] = self.molecular_train[col].astype(dtype)
                self.molecular_test[col] = self.molecular_test[col].astype(dtype)
        elif data_type == "cytogenetic":
            for col, dtype in type_dict.items():
                self.cytogenetic_train[col] = self.cytogenetic_train[col].astype(dtype)
                self.cytogenetic_test[col] = self.cytogenetic_test[col].astype(dtype)
        else:
            raise ValueError(f"Invalid data_type: {data_type}")
        
        return list(type_dict.keys())

    def encode_risk(
            self,
            categorical_cols: List[str],
            data_type: Literal["molecular", "clinical", "cytogenetic"]
            ) -> List[str]:
        """Encode risk categories for specified columns in the specified dataframes.

        Args:
            categorical_cols: list of column names to encode.
            data_type: the type of data to encode ("molecular", "clinical", or "cytogenetic").

        Returns:
            List of new feature names (strings).
        """
        # Select the appropriate dataframe
        if data_type == "clinical":
            df_train = self.clinical_train
            df_test = self.clinical_test
        elif data_type == "molecular":
            df_train = self.molecular_train
            df_test = self.molecular_test
        elif data_type == "cytogenetic":
            df_train = self.cytogenetic_train
            df_test = self.cytogenetic_test
        else:
            raise ValueError(f"Invalid data_type: {data_type}")

        # Validate input columns
        missing_cols = [c for c in categorical_cols if c not in df_train.columns]
        if missing_cols:
            raise KeyError(f"Columns missing from {data_type}_train: {missing_cols}")

        # Map risk categories to numerical values
        risk_mapping = {
            "very low": 0,
            "low": 1,
            "intermediate": 2,
            "high": 3,
            "very high": 4
        }
        for col in categorical_cols:
            df_train[col] = df_train[col].map(risk_mapping)
            df_test[col] = df_test[col].map(risk_mapping)

        return categorical_cols

    def one_hot_encode(
            self,
            categorical_cols: List[str],
            data_type: Literal["molecular", "clinical", "cytogenetic"]
            ) -> List[str]:
        """One-hot encode categorical columns and append new dummy columns to the specified dataframes.

        The encoder drops the first category (drop='first') and ignores unknown categories
        in the test set. The function returns the list of newly added feature names.

        Args:
            categorical_cols: list of column names to one-hot encode.
            data_type: the type of data to encode ("molecular", "clinical", or "cytogenetic").

        Returns:
            List of new feature names (strings).
        """
        
        # Select the appropriate dataframes
        if data_type == "clinical":
            df_train = self.clinical_train
            df_test = self.clinical_test
        elif data_type == "molecular":
            df_train = self.molecular_train
            df_test = self.molecular_test
        elif data_type == "cytogenetic":
            df_train = self.cytogenetic_train
            df_test = self.cytogenetic_test
        else:
            raise ValueError(f"Invalid data_type: {data_type}")

        # Validate input columns
        missing_in_train = [c for c in categorical_cols if c not in df_train.columns]
        missing_in_test = [c for c in categorical_cols if c not in df_test.columns]

        if missing_in_train:
            raise KeyError(f"Columns missing from {data_type}_train: {missing_in_train}")
        if missing_in_test:
            raise KeyError(f"Columns missing from {data_type}_test: {missing_in_test}")

        # Fit encoder on training data only. Use drop='first' to avoid collinearity.
        encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
        encoded_data_train = encoder.fit_transform(df_train[categorical_cols])
        new_feature_names = list(encoder.get_feature_names_out(categorical_cols))

        # If any of the new features already exist, drop them first to make operation idempotent
        overlapping = set(new_feature_names).intersection(df_train.columns)
        if overlapping:
            df_train = df_train.drop(columns=list(overlapping), errors='ignore')

        X_train_encoded = pd.DataFrame(encoded_data_train, columns=new_feature_names, index=df_train.index)
        df_train = pd.concat([df_train, X_train_encoded], axis=1)

        # Transform test; columns will match new_feature_names (unknown categories ignored)
        encoded_data_test = encoder.transform(df_test[categorical_cols])

        overlapping_test = set(new_feature_names).intersection(df_test.columns)
        if overlapping_test:
            df_test = df_test.drop(columns=list(overlapping_test), errors='ignore')

        X_test_encoded = pd.DataFrame(encoded_data_test, columns=new_feature_names, index=df_test.index)
        df_test = pd.concat([df_test, X_test_encoded], axis=1)

        # Update the instance variables
        if data_type == "clinical":
            self.clinical_train = df_train
            self.clinical_test = df_test
        elif data_type == "molecular":
            self.molecular_train = df_train
            self.molecular_test = df_test
        elif data_type == "cytogenetic":
            self.cytogenetic_train = df_train
            self.cytogenetic_test = df_test

        print(f"{len(new_feature_names)} new features added to {data_type} data.")

        return new_feature_names
    
    
    
    def Nmut(self) -> List[str]:
        """
        Crée un feature basé sur le nombre de mutations par patient (Nmut)

        Returns:
            List[str]: Liste des nouveaux features créés
        """
        # Step: Extract the number of somatic mutations per patient
        # Group by 'ID' and count the number of mutations (rows) per patient
        tmp = self.molecular_train.groupby('ID').size().reset_index(name='Nmut')

        # Drop existing Nmut column if present to ensure idempotency
        self.clinical_train = self.clinical_train.drop(columns=['Nmut'], errors='ignore')

        # Merge with the training dataset and replace missing values in 'Nmut' with 0
        self.clinical_train = self.clinical_train.merge(tmp, on='ID', how='left').fillna({'Nmut': 0})

        tmp = self.molecular_test.groupby('ID').size().reset_index(name='Nmut')

        self.clinical_test = self.clinical_test.drop(columns=['Nmut'], errors='ignore')

        self.clinical_test = self.clinical_test.merge(tmp, on='ID', how='left').fillna({'Nmut': 0})

        return ['Nmut']
    

    def ratios_and_interactions(self) -> List[str]:
        """
        Crée les ratios et interactions recommandés
        - WBC/ANC (proportion de cellules anormales)
        - PLT/HB (sévérité de l'atteinte médullaire)
        - BM_BLAST × complexité cytogénétique
        - Charge tumorale composite : (BM_BLAST + WBC_log) * risque_cyto

        Returns:
            List[str]: Liste des nouveaux features créés
        """

        # Features to create
        ratio_features = ['wbc_anc_ratio', 'plt_hb_ratio', 'blast_cyto_complexity', 'tumor_burden_composite']

        # Drop existing features to ensure idempotency
        self.clinical_train = self.clinical_train.drop(columns=ratio_features, errors='ignore')
        self.clinical_test = self.clinical_test.drop(columns=ratio_features, errors='ignore')

        # WBC/ANC (proportion de cellules anormales)
        self.clinical_train['wbc_anc_ratio'] = self.clinical_train['WBC'] / self.clinical_train['ANC']
        self.clinical_test['wbc_anc_ratio'] = self.clinical_test['WBC'] / self.clinical_test['ANC']

        # PLT/HB (sévérité de l'atteinte médullaire)
        self.clinical_train['plt_hb_ratio'] = self.clinical_train['PLT'] / self.clinical_train['HB']
        self.clinical_test['plt_hb_ratio'] = self.clinical_test['PLT'] / self.clinical_test['HB']

        # BM_BLAST × complexité cytogénétique
        self.clinical_train['blast_cyto_complexity'] = self.clinical_train['BM_BLAST'] * self.clinical_train['n_abnormalities']
        self.clinical_test['blast_cyto_complexity'] = self.clinical_test['BM_BLAST'] * self.clinical_test['n_abnormalities']

        # Charge tumorale composite: (BM_BLAST + WBC_log) × risque_cyto
        wbc_log_train = np.log1p(self.clinical_train['WBC'])
        wbc_log_test = np.log1p(self.clinical_test['WBC'])

        # Convertir risque cyto en numérique pour le calcul (utiliser computed_risk_score)
        cyto_risk_num_train = self.clinical_train['computed_risk_score']
        cyto_risk_num_test = self.clinical_test['computed_risk_score']

        self.clinical_train['tumor_burden_composite'] = (self.clinical_train['BM_BLAST'] + wbc_log_train) * cyto_risk_num_train
        self.clinical_test['tumor_burden_composite'] = (self.clinical_test['BM_BLAST'] + wbc_log_test) * cyto_risk_num_test

        return ratio_features
    
    def random(self, seed: Optional[int] = None) -> List[str]:
        """
        Crée un feature aléatoire pour test

        Returns:
            List[str]: Liste des nouveaux features créés
        """
        rng = np.random.default_rng(seed)

        # Features to create
        random_features = ['random_feature']

        # Drop existing features to ensure idempotency
        self.clinical_train = self.clinical_train.drop(columns=random_features, errors='ignore')
        self.clinical_test = self.clinical_test.drop(columns=random_features, errors='ignore')

        # Create random feature
        self.clinical_train['random_feature'] = rng.normal(size=len(self.clinical_train))
        self.clinical_test['random_feature'] = rng.normal(size=len(self.clinical_test))

        return random_features

    def severity_scores(self) -> List[str]:
        """
        Crée les scores de sévérité globale
        - Nombre de cytopénies (HB bas + PLT bas + ANC bas)

        Returns:
            List[str]: Liste des nouveaux features créés
        """
        # Features to create
        severity_features = ['cytopenias_count']

        # Drop existing features to ensure idempotency
        self.clinical_train = self.clinical_train.drop(columns=severity_features, errors='ignore')
        self.clinical_test = self.clinical_test.drop(columns=severity_features, errors='ignore')

        # Nombre de cytopénies (HB < 10 + PLT < 100 + ANC < 1.5)
        def count_cytopenias(row):
            count = 0
            if row['HB'] < 10:
                count += 1
            if row['PLT'] < 100:
                count += 1
            if row['ANC'] < 1.5:
                count += 1
            return count

        self.clinical_train['cytopenias_count'] = self.clinical_train.apply(count_cytopenias, axis=1)
        self.clinical_test['cytopenias_count'] = self.clinical_test.apply(count_cytopenias, axis=1)

        return severity_features
    
    
    
    @staticmethod
    def _map_gene_to_pathway():
        """
        Retourne une fonction qui mappe un gène à son pathway.
        """
        def mapper(gene: str) -> str:
            for pathway, genes in PATHWAY_GENES.items():
                if gene in genes:
                    return pathway.value
            return 'OTHER'
        return mapper

    def _pathways_classification(
        self,
    )  -> None:
        """
        Ajoute au dataframe molecular une colonne de pathways mutés par patient.
        """
    
        # Créer une colonne binaire indiquant si le patient a une mutation dans ce pathway
        for df in [self.molecular_train, self.molecular_test]:
            df['PATHWAY'] = df['GENE'].apply(FeatureEngineer2._map_gene_to_pathway())

    @staticmethod
    def _create_confidence_weighted_count_matrix(
        col: str,
        method: Literal['confidence_weighted', 'bayesian', 'vaf_score', 'log_vaf', 'depth_score', 'constant'],
        apply_effect_weighting: bool,
        cat_to_include: Optional[List[str]],
        df_molecular: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Crée une matrice patient × gènes avec comptage pondéré par VAF sigmoid
        
        Pour chaque patient et chaque gène:
        - Si aucune mutation: 0
        - Si 1 mutation: confidence_weighted_vaf(VAF, DEPTH)
        - Si N mutations: SOMME des confidence_weighted_vaf de toutes les mutations
        
        Args:
            df_molecular: DataFrame avec colonnes ID, GENE, VAF, DEPTH. Si None, utilise self.molecular_data_train.
            genes_to_include: Liste des gènes à inclure (None = tous les gènes fréquents)
            min_patient_frequency: Nombre minimum de patients avec mutation (filtre si genes_to_include=None)
        
        Returns:
            DataFrame avec index=ID, colonnes=gènes, valeurs=score pondéré
        """

        if cat_to_include is None:
            # Déterminer les gènes fréquents
            cat_to_include = list(df_molecular[col].unique())

        # Filtrer les gènes
        df_filtered = df_molecular[df_molecular[col].isin(cat_to_include)].copy()

        vaf_mean = df_filtered['VAF'].mean()
        depth_mean = df_filtered['DEPTH'].mean()

        wvaf_calculator = GeneEnhancedEncoder(
            method=method, 
            vaf_mean=vaf_mean, 
            depth_mean=depth_mean,
            apply_effect_weighting=apply_effect_weighting
            )

        # Calculer le score pondéré pour chaque mutation
        df_filtered['weighted_score'] = df_filtered.apply(
            lambda row: wvaf_calculator.compute(row['VAF'], row['DEPTH'], row['EFFECT']),
            axis=1
        )

        # Agréger par patient × gène (SOMME des scores pondérés)
        gene_matrix = (df_filtered.groupby(['ID', col])['weighted_score']
                       .sum()
                       .unstack(fill_value=0))

        # S'assurer que tous les patients sont présents (même ceux sans mutations)
        all_patients = df_molecular['ID'].unique()
        gene_matrix = gene_matrix.reindex(all_patients, fill_value=0)

        return gene_matrix


    def add_mol_encoding(
        self,
        col: str,
        method: Literal['confidence_weighted', 'bayesian', 'vaf_score', 'log_vaf', 'depth_score', 'constant'],
        apply_effect_weighting: bool,
        cat_to_include: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Construis la matrice pondérée (si nécessaire), merge les colonnes gènes dans
        `self.clinical_train` et `self.clinical_test`, et renvoie la liste des nouvelles colonnes ajoutées.

        Comportement:
        - Calcule la matrice pondérée pour training et test (utilise les molecular_train/test)
        - Aligne les colonnes (union des gènes) et remplit les valeurs manquantes par 0
        - Merge (left) sur la colonne 'ID' et garantit l'idempotence en supprimant
          d'abord les colonnes existantes portant les mêmes noms.

        Args:
            genes_to_include: liste optionnelle de gènes à inclure. Si None, utilise
                              le filtrage par `min_patient_frequency`.
            min_patient_frequency: si `genes_to_include` est None, nombre min de patients
                                   pour qu'un gène soit conservé.

        Returns:
            List[str]: noms des colonnes gènes ajoutées dans `self.clinical_train`/`self.clinical_test`.
        """

        if col == "PATHWAY" and  'PATHWAY' not in self.molecular_train.columns:
            self._pathways_classification()

        if cat_to_include is None:
            cat_to_include = list(self.molecular_test[col].unique())


        # Calculer matrices pour train et test
        train_mat = self._create_confidence_weighted_count_matrix(
            col=col, 
            method=method,
            cat_to_include=cat_to_include, 
            apply_effect_weighting=apply_effect_weighting,
            df_molecular=self.molecular_train
            )

        test_mat = self._create_confidence_weighted_count_matrix(
            col=col, 
            method=method,
            cat_to_include=cat_to_include, 
            apply_effect_weighting=apply_effect_weighting,
            df_molecular=self.molecular_test
            )

        new_col_names = []
        for cat in cat_to_include:
            cat_s = cat
            suffix = f"{method}"
            if apply_effect_weighting:
                suffix = f"{suffix}__effect"
            new_name = f"{cat_s}__{suffix}"
            new_col_names.append(new_name)

        # Renommer colonnes des matrices (mapping old_name -> new_name)
        rename_map = dict(zip(cat_to_include, new_col_names))
        train_mat = train_mat.rename(columns=rename_map)
        test_mat = test_mat.rename(columns=rename_map)

        # Aligner colonnes (utiliser la liste de noms final)
        train_mat = train_mat.reindex(columns=new_col_names, fill_value=0)
        test_mat = test_mat.reindex(columns=new_col_names, fill_value=0)

        # Préparer pour merge: reset index pour obtenir 'ID' comme colonne
        train_df_for_merge = train_mat.reset_index()
        test_df_for_merge = test_mat.reset_index()

        # Supprimer d'éventuelles colonnes déjà présentes (idempotence) — on supprime par nouveaux noms
        self.clinical_train = self.clinical_train.drop(columns=new_col_names, errors='ignore')
        self.clinical_test  = self.clinical_test.drop(columns=new_col_names, errors='ignore')

        # Merge (left) — si certains patients n'ont pas de ligne dans la matrice, on remplira par 0
        new_clinical_train = self.clinical_train.merge(train_df_for_merge, on='ID', how='left')
        new_clinical_test  = self.clinical_test.merge(test_df_for_merge, on='ID', how='left')

        mask = new_clinical_train.isna()
        new_clinical_train[mask] = 0
        mask_test = new_clinical_test.isna()
        new_clinical_test[mask_test] = 0
        self.clinical_train = new_clinical_train
        self.clinical_test  = new_clinical_test

        print(f"Ajoutées {len(new_col_names)} colonnes encodées pour '{col}' (method={method}, apply_effect_weighting={apply_effect_weighting}).")

        return new_col_names