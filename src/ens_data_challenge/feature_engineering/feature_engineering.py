from typing import Any, List, Optional, Literal
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import OneHotEncoder

from ens_data_challenge.types import HighRiskGenes, PATHWAY_GENES
from .utils.genes_coefs_calculator import GeneEnhancedEncoder

class FeatureEngineering:
    def __init__(
            self, 
            clinical_data_train: pd.DataFrame, 
            clinical_data_test: pd.DataFrame,
            molecular_data_train: pd.DataFrame,
            molecular_data_test: pd.DataFrame,
            target_data: pd.DataFrame
            ) :
        
        self.X_train = clinical_data_train.copy()
        self.X_test = clinical_data_test.copy()

        self.molecular_data_train = molecular_data_train
        self.molecular_data_test = molecular_data_test
        self.target_data = target_data

    def get_X_train(self) -> pd.DataFrame:
        return self.X_train
    
    def get_X_test(self) -> pd.DataFrame:
        return self.X_test
    
    def as_type(self, cols: List[str], dtype: Any) -> None:
        """Convert column to specified data type in both X_train and X_test."""
        self.X_train[cols] = self.X_train[cols].astype(dtype)
        self.X_test[cols] = self.X_test[cols].astype(dtype)

    
    def encode_categorical(
            self,
            categorical_cols: List[str]
            ) -> List[str]:
        """One-hot encode categorical columns and append new dummy columns to X_train/X_test.

        The encoder drops the first category (drop='first') and ignores unknown categories
        in the test set. The function returns the augmented train and test DataFrames
        plus a list of the newly added feature names.

        Args:
            categorical_cols: list of column names to one-hot encode.
            X_train: training DataFrame containing categorical_cols.
            X_test: test DataFrame containing categorical_cols.

        Returns:
            Tuple containing:
              - X_train augmented with one-hot columns,
              - X_test augmented with one-hot columns,
              - list of new feature names (strings).
        """
        
        # Validate input columns
        missing_in_train = [c for c in categorical_cols if c not in self.X_train.columns]
        missing_in_test = [c for c in categorical_cols if c not in self.X_test.columns]

        if missing_in_train:
            raise KeyError(f"Columns missing from X_train: {missing_in_train}")
        if missing_in_test:
            raise KeyError(f"Columns missing from X_test: {missing_in_test}")

        # Fit encoder on training data only. Use drop='first' to avoid collinearity.
        encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
        encoded_data_train = encoder.fit_transform(self.X_train[categorical_cols])
        new_feature_names = list(encoder.get_feature_names_out(categorical_cols))

        # If any of the new features already exist, drop them first to make operation idempotent
        overlapping = set(new_feature_names).intersection(self.X_train.columns)
        if overlapping:
            self.X_train = self.X_train.drop(columns=list(overlapping), errors='ignore')

        X_train_encoded = pd.DataFrame(encoded_data_train, columns=new_feature_names, index=self.X_train.index)
        self.X_train = pd.concat([self.X_train, X_train_encoded], axis=1)

        # Transform test; columns will match new_feature_names (unknown categories ignored)
        encoded_data_test = encoder.transform(self.X_test[categorical_cols])

        overlapping_test = set(new_feature_names).intersection(self.X_test.columns)
        if overlapping_test:
            self.X_test = self.X_test.drop(columns=list(overlapping_test), errors='ignore')

        X_test_encoded = pd.DataFrame(encoded_data_test, columns=new_feature_names, index=self.X_test.index)
        self.X_test = pd.concat([self.X_test, X_test_encoded], axis=1)

        print(f"{len(new_feature_names)} new features added.")

        return new_feature_names
    
    
    
    def Nmut(self) -> List[str]:
        """
        Crée un feature basé sur le nombre de mutations par patient (Nmut)

        Returns:
            List[str]: Liste des nouveaux features créés
        """
        # Step: Extract the number of somatic mutations per patient
        # Group by 'ID' and count the number of mutations (rows) per patient
        tmp = self.molecular_data_train.groupby('ID').size().reset_index(name='Nmut')

        # Drop existing Nmut column if present to ensure idempotency
        self.X_train = self.X_train.drop(columns=['Nmut'], errors='ignore')

        # Merge with the training dataset and replace missing values in 'Nmut' with 0
        self.X_train = self.X_train.merge(tmp, on='ID', how='left').fillna({'Nmut': 0})

        tmp = self.molecular_data_test.groupby('ID').size().reset_index(name='Nmut')

        self.X_test = self.X_test.drop(columns=['Nmut'], errors='ignore')

        self.X_test = self.X_test.merge(tmp, on='ID', how='left').fillna({'Nmut': 0})

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
        self.X_train = self.X_train.drop(columns=ratio_features, errors='ignore')
        self.X_test = self.X_test.drop(columns=ratio_features, errors='ignore')

        # WBC/ANC (proportion de cellules anormales)
        self.X_train['wbc_anc_ratio'] = self.X_train['WBC'] / self.X_train['ANC']
        self.X_test['wbc_anc_ratio'] = self.X_test['WBC'] / self.X_test['ANC']

        # PLT/HB (sévérité de l'atteinte médullaire)
        self.X_train['plt_hb_ratio'] = self.X_train['PLT'] / self.X_train['HB']
        self.X_test['plt_hb_ratio'] = self.X_test['PLT'] / self.X_test['HB']

        # BM_BLAST × complexité cytogénétique
        self.X_train['blast_cyto_complexity'] = self.X_train['BM_BLAST'] * self.X_train['n_abnormalities']
        self.X_test['blast_cyto_complexity'] = self.X_test['BM_BLAST'] * self.X_test['n_abnormalities']

        # Charge tumorale composite: (BM_BLAST + WBC_log) × risque_cyto
        wbc_log_train = np.log1p(self.X_train['WBC'])
        wbc_log_test = np.log1p(self.X_test['WBC'])

        # Convertir risque cyto en numérique pour le calcul (utiliser computed_risk_score)
        cyto_risk_num_train = self.X_train['computed_risk_score']
        cyto_risk_num_test = self.X_test['computed_risk_score']

        self.X_train['tumor_burden_composite'] = (self.X_train['BM_BLAST'] + wbc_log_train) * cyto_risk_num_train
        self.X_test['tumor_burden_composite'] = (self.X_test['BM_BLAST'] + wbc_log_test) * cyto_risk_num_test

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
        self.X_train = self.X_train.drop(columns=random_features, errors='ignore')
        self.X_test = self.X_test.drop(columns=random_features, errors='ignore')

        # Create random feature
        self.X_train['random_feature'] = rng.normal(size=len(self.X_train))
        self.X_test['random_feature'] = rng.normal(size=len(self.X_test))

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
        self.X_train = self.X_train.drop(columns=severity_features, errors='ignore')
        self.X_test = self.X_test.drop(columns=severity_features, errors='ignore')

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

        self.X_train['cytopenias_count'] = self.X_train.apply(count_cytopenias, axis=1)
        self.X_test['cytopenias_count'] = self.X_test.apply(count_cytopenias, axis=1)

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
        for df in [self.molecular_data_train, self.molecular_data_test]:
            df['PATHWAY'] = df['GENE'].apply(FeatureEngineering._map_gene_to_pathway())


    def _create_confidence_weighted_count_matrix(
        self,
        col: str,
        method: Literal['confidence_weighted', 'bayesian', 'vaf_score', 'log_vaf', 'depth_score', 'constant'],
        apply_effect_weighting: bool,
        cat_to_include: Optional[List[str]],
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
        # Use provided DataFrame or fallback to training molecular data
        df_molecular = self.molecular_data_train

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
        `self.X_train` et `self.X_test`, et renvoie la liste des nouvelles colonnes ajoutées.

        Comportement:
        - Calcule la matrice pondérée pour training et test (utilise les molecular_data_train/test)
        - Aligne les colonnes (union des gènes) et remplit les valeurs manquantes par 0
        - Merge (left) sur la colonne 'ID' et garantit l'idempotence en supprimant
          d'abord les colonnes existantes portant les mêmes noms.

        Args:
            genes_to_include: liste optionnelle de gènes à inclure. Si None, utilise
                              le filtrage par `min_patient_frequency`.
            min_patient_frequency: si `genes_to_include` est None, nombre min de patients
                                   pour qu'un gène soit conservé.

        Returns:
            List[str]: noms des colonnes gènes ajoutées dans `self.X_train`/`self.X_test`.
        """

        if col == "PATHWAY" and  'PATHWAY' not in self.molecular_data_train.columns:
            self._pathways_classification()

        if cat_to_include is None:
            cat_to_include = list(self.molecular_data_test[col].unique())


        # Calculer matrices pour train et test
        train_mat = self._create_confidence_weighted_count_matrix(
            col=col, 
            method=method,
            cat_to_include=cat_to_include, 
            apply_effect_weighting=apply_effect_weighting
            )

        test_mat = self._create_confidence_weighted_count_matrix(
            col=col, 
            method=method,
            cat_to_include=cat_to_include, 
            apply_effect_weighting=apply_effect_weighting
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
        self.X_train = self.X_train.drop(columns=new_col_names, errors='ignore')
        self.X_test  = self.X_test.drop(columns=new_col_names, errors='ignore')

        # Merge (left) — si certains patients n'ont pas de ligne dans la matrice, on remplira par 0
        self.X_train = self.X_train.merge(train_df_for_merge, on='ID', how='left')
        self.X_test  = self.X_test.merge(test_df_for_merge, on='ID', how='left')

        print(f"Ajoutées {len(new_col_names)} colonnes encodées pour '{col}' (method={method}, apply_effect_weighting={apply_effect_weighting}).")

        return new_col_names