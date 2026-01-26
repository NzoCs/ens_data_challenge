from typing import Any, Dict, List, Optional, Literal
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA

from ens_data_challenge.types import PATHWAY_GENES
from .utils.genes_coefs_calculator import GeneEnhancedEncoder


class FeatureEngineerHelper:
    def __init__(self):
        self.one_hot_encoder = OneHotEncoder(sparse_output=False, drop="first")
        self.pcas = {}
        self.encoding_categories = {}

    def as_type(self, data: pd.DataFrame, type_dict: Dict[str, Any]) -> pd.DataFrame:
        """Convert column to specified data type in the specified dataframes."""
        # Select the appropriate dataframe

        data = data.copy()
        for col, dtype in type_dict.items():
            if col not in data.columns:
                raise KeyError(f"Column '{col}' not found in dataframe.")
            data[col] = data[col].astype(dtype)

        return data

    def one_hot_encode_fit(
        self,
        train_data: pd.DataFrame,
        categorical_cols: List[str],
    ) -> None:
        """Fits the one_hot encoder in the self, needs to be applied to the train data, then transformed on test data and train data.

        The encoder drops the first category (drop='first') and ignores unknown categories
        in the test set. The function returns the list of newly added feature names.

        Args:
            train_data: training dataframe.
            categorical_cols: list of column names to one-hot encode.
        """

        self.one_hot_encoder = OneHotEncoder(sparse_output=False, drop="first")

        self.one_hot_encoder.fit(train_data[categorical_cols])

    def one_hot_encode_transform(
        self,
        data: pd.DataFrame,
        categorical_cols: List[str],
    ) -> pd.DataFrame:
        """Applies the fitted one_hot encoder to the specified dataframe.
        Args:
            data: dataframe to transform.
            categorical_cols: list of column names to one-hot encode.
        Returns:
            Transformed dataframe and list of new feature names (strings).
        """

        df = data.copy()
        # Validate input columns

        missing_cols = [c for c in categorical_cols if c not in df.columns]
        if missing_cols:
            raise KeyError(f"Columns missing from dataframe: {missing_cols}")
        # Apply one-hot encoding

        ohe_array = self.one_hot_encoder.transform(df[categorical_cols])

        ohe_feature_names = self.one_hot_encoder.get_feature_names_out(categorical_cols)
        ohe_df = pd.DataFrame(data=ohe_array, columns=ohe_feature_names, index=df.index)  # type: ignore

        # Drop original categorical columns and concatenate one-hot encoded columns
        df = df.drop(columns=categorical_cols)
        df = pd.concat([df, ohe_df], axis=1)

        return df

    def Nmut(
        self, molecular_data: pd.DataFrame, clinical_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Crée un feature basé sur le nombre de mutations par patient (Nmut)

        Args:
            molecular_data (pd.DataFrame): DataFrame moléculaire avec colonne 'ID'
            clinical_data (pd.DataFrame): DataFrame clinique avec colonne 'ID'

        Returns:
            clinical_data avec la colonne 'Nmut' ajoutée
        """
        # Step: Extract the number of somatic mutations per patient
        # Group by 'ID' and count the number of mutations (rows) per patient
        tmp = molecular_data.groupby("ID").size().reset_index(name="Nmut")  # type: ignore

        # Drop existing Nmut column if present to ensure idempotency
        clinical_data = clinical_data.drop(columns=["Nmut"], errors="ignore")

        # Merge with the training dataset and replace missing values in 'Nmut' with 0
        clinical_data = clinical_data.merge(tmp, on="ID", how="left").fillna(
            {"Nmut": 0}
        )

        tmp = molecular_data.groupby("ID").size().reset_index(name="Nmut")  # type: ignore

        clinical_data = clinical_data.drop(columns=["Nmut"], errors="ignore")

        clinical_data = clinical_data.merge(tmp, on="ID", how="left").fillna(
            {"Nmut": 0}
        )

        return clinical_data

    def ratios_and_interactions(self, clinical_data: pd.DataFrame) -> pd.DataFrame:
        """
        Crée les ratios et interactions recommandés
        - WBC/ANC (proportion de cellules anormales)
        - PLT/HB (sévérité de l'atteinte médullaire)
        - BM_BLAST × complexité cytogénétique
        - Charge tumorale composite : (BM_BLAST + WBC_log) * risque_cyto

        Args:
            clinical_data (pd.DataFrame): DataFrame clinique avec les colonnes nécessaires pour le calcul

        Returns:
            DataFrame with the new ratio and interaction features added
        """

        clinical_data = clinical_data.copy()

        # WBC/ANC (proportion de cellules anormales)
        clinical_data["wbc_anc_ratio"] = clinical_data["WBC"] / clinical_data["ANC"]
        # remplacer les inf par 0 (si ANC=0)
        clinical_data["wbc_anc_ratio"].replace([np.inf, -np.inf], 0, inplace=True)

        clinical_data["plt_hb_ratio"] = clinical_data["PLT"] / clinical_data["HB"]
        # remplacer les inf par 0 (si HB=0)
        clinical_data["plt_hb_ratio"].replace([np.inf, -np.inf], 0, inplace=True)

        # BLAST × complexité cytogénétique
        clinical_data["blast_cyto_complexity"] = (
            clinical_data["BM_BLAST"] * clinical_data["n_abnormalities"]
        )
        # Charge tumorale composite: (BM_BLAST + WBC_log) × risque_cyto
        wbc_log = np.log1p(clinical_data["WBC"])

        # Convertir risque cyto en numérique pour le calcul (utiliser computed_risk_score)
        cyto_risk_num = clinical_data["computed_risk_score"]

        clinical_data["tumor_burden_composite"] = (
            clinical_data["BM_BLAST"] + wbc_log
        ) * cyto_risk_num

        return clinical_data

    def random(self, data: pd.DataFrame, seed: Optional[int] = None) -> pd.DataFrame:
        """
        Crée un feature aléatoire pour test

        Args:
            data (pd.DataFrame): DataFrame to add the random feature to
            seed (Optional[int]): Seed pour la reproductibilité

        Returns:
            pd.DataFrame: DataFrame with the new random feature added
        """
        rng = np.random.default_rng(seed)

        # Features to create
        random_features = ["random_feature"]

        # Drop existing features to ensure idempotency
        data = data.drop(columns=random_features, errors="ignore")

        # Create random feature
        data["random_feature"] = rng.normal(size=len(data))

        return data

    def severity_scores(self, clinical_data: pd.DataFrame) -> pd.DataFrame:
        """
        Crée les scores de sévérité globale
        - Nombre de cytopénies (HB bas + PLT bas + ANC bas)

        Args:
            clinical_data (pd.DataFrame): DataFrame clinique avec les colonnes nécessaires pour le calcul (HB, PLT, ANC)
        Returns:
            DataFrame with the new severity features added
        """
        clinical_data = clinical_data.copy()

        # Nombre de cytopénies (HB < 10 + PLT < 100 + ANC < 1.5)
        def count_cytopenias(row):
            count = 0
            if row["HB"] < 10:
                count += 1
            if row["PLT"] < 100:
                count += 1
            if row["ANC"] < 1.5:
                count += 1
            return count

        clinical_data["cytopenias_count"] = clinical_data.apply(
            count_cytopenias, axis=1
        )

        return clinical_data

    @staticmethod
    def _map_gene_to_pathway():
        """
        Retourne une fonction qui mappe un gène à son pathway.
        """

        def mapper(gene: str) -> str:
            for pathway, genes in PATHWAY_GENES.items():
                if gene in genes:
                    return pathway.value
            return "OTHER"

        return mapper

    def _pathways_classification(
        self,
        molecular_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Ajoute au dataframe molecular une colonne de pathways mutés par patient.

        Args:
            molecular_data (pd.DataFrame): DataFrame moléculaire avec colonne 'GENE'
        Returns:
            pd.DataFrame: DataFrame moléculaire avec colonne 'PATHWAY' added
        """

        molecular_data = molecular_data.copy()

        # Créer une colonne binaire indiquant si le patient a une mutation dans ce pathway
        molecular_data["PATHWAY"] = molecular_data["GENE"].apply(
            FeatureEngineerHelper._map_gene_to_pathway()
        )

        return molecular_data

    @staticmethod
    def _create_confidence_weighted_count_matrix(
        col: str,
        molecular_data: pd.DataFrame,
        method: Literal[
            "confidence_weighted",
            "bayesian",
            "vaf_score",
            "log_vaf",
            "depth_score",
            "constant",
        ],
        apply_effect_weighting: bool,
        categories_to_include: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Crée une matrice pondérée de gènes (ou pathways) par patient en utilisant la méthode
        de score pondéré spécifiée.

        Pour chaque patient et chaque gène:
        - Si aucune mutation: 0
        - Si 1 mutation: confidence_weighted_vaf(VAF, DEPTH)
        - Si N mutations: SOMME des confidence_weighted_vaf de toutes les mutations

        Args:
            molecular_data: DataFrame avec colonnes ID, GENE, VAF, DEPTH. Si None, utilise self.molecular_data_train.
            col: colonne à utiliser pour les gènes (ex: 'GENE' ou 'PATHWAY')
            categories_to_include: liste optionnelle de gènes à inclure. Si None, utilise
                                   tous les gènes présents dans molecular_data.
            method: méthode de calcul du score pondéré
            apply_effect_weighting: si True, applique un poids basé sur l'effet de la mutation

        Returns:
            DataFrame avec index=ID, colonnes=gènes, valeurs=score pondéré
        """

        if categories_to_include is None:
            categories_to_include = list(molecular_data[col].unique())

        # Filtrer les gènes
        # On selectionne les lignes où la colonne col est dans categories_to_include
        df_filtered = molecular_data[
            molecular_data[col].isin(categories_to_include)
        ].copy()

        vaf_mean = df_filtered["VAF"].mean()
        depth_mean = df_filtered["DEPTH"].mean()

        wvaf_calculator = GeneEnhancedEncoder(
            method=method,
            vaf_mean=vaf_mean,
            depth_mean=depth_mean,
            apply_effect_weighting=apply_effect_weighting,
        )

        assert df_filtered.index.is_unique is True

        # Calculer le score pondéré pour chaque mutation
        df_filtered["weighted_score"] = df_filtered.apply(
            lambda row: wvaf_calculator.compute(
                row["VAF"], row["DEPTH"], row["EFFECT"]
            ),
            axis=1,
        )

        # Agréger par patient × gène (SOMME des scores pondérés)
        gene_matrix = (
            df_filtered.groupby(["ID", col])["weighted_score"]
            .sum()
            .unstack(fill_value=0)
        )

        # S'assurer que tous les patients sont présents (même ceux sans mutations)
        all_patients = molecular_data["ID"].unique()
        gene_matrix = gene_matrix.reindex(all_patients, fill_value=0)

        return gene_matrix

    def mol_encoding(
        self,
        molecular_data: pd.DataFrame,
        col: Literal["GENE", "PATHWAY"],
        method: Literal[
            "confidence_weighted",
            "bayesian",
            "vaf_score",
            "log_vaf",
            "depth_score",
            "constant",
        ],
        apply_effect_weighting: bool,
        categories_to_include: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Ajoute des colonnes encodées basées sur les données moléculaires au DataFrame clinique spécifié.

        Args:
            clinical_data: DataFrame clinique (train ou test) avec colonne 'ID'
            molecular_data: DataFrame moléculaire avec colonnes ID, GENE, VAF, DEPTH. Si None, utilise self.molecular_train et self.molecular_test.
            categories_to_include: liste optionnelle de catégories à inclure. Si None, utilise
                                   le filtrage par `min_patient_frequency`.
            col: colonne à utiliser pour les gènes (ex: 'GENE' ou 'PATHWAY')
            method: méthode de calcul du score pondéré
            apply_effect_weighting: si True, applique un poids basé sur l'effet de la mutation

        Returns:
            pd.DataFrame: Matrice des gènes pondérés par patient
        """
        molecular_data = molecular_data.copy()

        if col == "PATHWAY" and "PATHWAY" not in molecular_data.columns:
            molecular_data = self._pathways_classification(molecular_data)

        if categories_to_include is None:
            categories_to_include = list(molecular_data[col].unique())

        # Calculer matrices pour train et test
        molecular_data = self._create_confidence_weighted_count_matrix(
            col=col,
            method=method,
            categories_to_include=categories_to_include,
            apply_effect_weighting=apply_effect_weighting,
            molecular_data=molecular_data,
        )
        # Create column names
        new_col_names = []
        for cat in categories_to_include:
            cat_s = cat
            suffix = f"{method}"
            if apply_effect_weighting:
                suffix = f"{suffix}__effect"
            new_name = f"{cat_s}__{suffix}"
            new_col_names.append(new_name)

        # Renommer colonnes des matrices (mapping old_name -> new_name)
        rename_map = dict(zip(categories_to_include, new_col_names))
        molecular_data = molecular_data.rename(columns=rename_map)

        # Aligner colonnes (utiliser la liste de noms final)
        molecular_data = molecular_data.reindex(columns=new_col_names, fill_value=0)

        return molecular_data

    def fit_pca(
        self,
        data: pd.DataFrame,
        n_components: Optional[int] = None,
        col: Literal["GENE", "PATHWAY"] = "GENE",
    ) -> None:
        """Fits PCA on the provided data."""
        pca = PCA(n_components=n_components)
        pca.fit(data)
        self.pcas[col] = pca
        print(f"PCA fitted for {col} with {pca.n_components_} components.")

    def transform_pca(
        self,
        data: pd.DataFrame,
        col: Literal["GENE", "PATHWAY"] = "GENE",
    ) -> pd.DataFrame:
        """Applies the fitted PCA to the provided data."""
        if col not in self.pcas:
            raise ValueError(f"PCA not fitted for {col}. Call fit_pca first.")

        pca = self.pcas[col]
        data_pca = pca.transform(data)

        new_col_names = [f"{col}_PCA_{i}" for i in range(pca.n_components_)]

        return pd.DataFrame(data_pca, columns=new_col_names, index=data.index)

    def merge_encoding(
        self,
        clinical_data: pd.DataFrame,
        molecular_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Merges the molecular encoding into the clinical data."""
        # Préparer pour merge: reset index pour obtenir 'ID' comme colonne
        df_for_merge = molecular_data.reset_index()

        # Merge (left) — si certains patients n'ont pas de ligne dans la matrice, on remplira par 0
        new_clinical_data = clinical_data.merge(df_for_merge, on="ID", how="left")

        mask = new_clinical_data.isna()
        new_clinical_data[mask] = 0
        new_clinical_data = new_clinical_data

        return new_clinical_data

    def fit_transform_mol_encoding(
        self,
        clinical_data: pd.DataFrame,
        molecular_data: pd.DataFrame,
        col: Literal["GENE", "PATHWAY"],
        method: Literal[
            "confidence_weighted",
            "bayesian",
            "vaf_score",
            "log_vaf",
            "depth_score",
            "constant",
        ],
        apply_effect_weighting: bool,
        categories_to_include: Optional[List[str]] = None,
        apply_pca: bool = True,
        n_pca_components: Optional[int] = None,
    ) -> pd.DataFrame:
        """Fits the encoding categories and optional PCA, then transforms and merges."""
        molecular_data = molecular_data.copy()

        # 1. Determine and store categories
        if col == "PATHWAY" and "PATHWAY" not in molecular_data.columns:
            molecular_data = self._pathways_classification(molecular_data)

        if categories_to_include is None:
            categories_to_include = list(molecular_data[col].unique())

        self.encoding_categories[col] = categories_to_include

        # 2. Compute encoding matrix
        mol_matrix = self.mol_encoding(
            molecular_data=molecular_data,
            col=col,
            method=method,
            apply_effect_weighting=apply_effect_weighting,
            categories_to_include=categories_to_include,
        )

        # 3. Fit and apply PCA if requested
        if apply_pca:
            self.fit_pca(mol_matrix, n_components=n_pca_components, col=col)
            mol_matrix = self.transform_pca(mol_matrix, col=col)
        elif col in self.pcas:
            del self.pcas[col]

        # 4. Merge
        return self.merge_encoding(clinical_data, mol_matrix)

    def transform_mol_encoding(
        self,
        clinical_data: pd.DataFrame,
        molecular_data: pd.DataFrame,
        col: Literal["GENE", "PATHWAY"],
        method: Literal[
            "confidence_weighted",
            "bayesian",
            "vaf_score",
            "log_vaf",
            "depth_score",
            "constant",
        ],
        apply_effect_weighting: bool,
    ) -> pd.DataFrame:
        """Transforms using stored categories and optional fitted PCA."""
        molecular_data = molecular_data.copy()

        # 1. Retrieve stored categories
        if col not in self.encoding_categories:
            # Fallback or error? Let's error to ensure strict usage
            raise ValueError(
                f"Encoding categories for {col} not found. Call fit_transform_mol_encoding first."
            )

        categories_to_include = self.encoding_categories[col]

        # 2. Compute encoding matrix
        mol_matrix = self.mol_encoding(
            molecular_data=molecular_data,
            col=col,
            method=method,
            apply_effect_weighting=apply_effect_weighting,
            categories_to_include=categories_to_include,
        )

        # 3. Apply PCA if it was fitted
        if col in self.pcas:
            mol_matrix = self.transform_pca(mol_matrix, col=col)

        # 4. Merge
        return self.merge_encoding(clinical_data, mol_matrix)
