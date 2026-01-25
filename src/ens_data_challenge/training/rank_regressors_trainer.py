from typing import Protocol
import pandas as pd

from ens_data_challenge.training.utils import get_ipcw_weights

class RankRegressor(Protocol):
    def train(self, data: pd.DataFrame, targets: pd.DataFrame, model: ModelWrapper):
        """Takes a preprocess, engineered DataFrame and a model, computes IPCW weights, and trains the model using these weights.
        
        Args:
            data: DataFrame with features
            targets: DataFrame with target columns
            model: ModelWrapper instance with fit method
        
        Returns: None
        """
        ...

class GlobalRankRegressor(RankRegressor):

    time_col: str
    event_col: str

    def __init__(self):
        self.time_col = "OS_YEARS"
        self.event_col = "OS_STATUS"

    def train(self, data: pd.DataFrame, targets: pd.DataFrame, model: ModelWrapper):


        # Exemple d'utilisation des poids IPCW dans l'entraînement
        weights_df = get_ipcw_weights(
            targets[self.time_col],
            targets[self.event_col],
        )

        # Utiliser les poids dans l'entraînement du modèle
        model.fit(
            X=data,
            y=targets[self.time_col],
            sample_weight=weights_df
        )

class DeadRankRegressor(RankRegressor):

    time_col: str
    event_col: str

    def __init__(self):
        self.time_col = "OS_YEARS"
        self.event_col = "OS_STATUS"

    def train(self, data: pd.DataFrame, targets: pd.DataFrame, model: ModelWrapper):

        """Trains the model only on uncensored (dead) patients.
        
        Args:
            data: DataFrame with features
            targets: DataFrame with target columns
            model: ModelWrapper instance with fit method
        
        Returns: None
        """

        # Filtrer pour ne garder que les patients décédés
        dead_targets = targets[targets[self.event_col] == 1]

        # Entraîner le modèle sur les patients décédés uniquement
        model.fit(
            X=data.loc[dead_targets.index],
            y=dead_targets[self.time_col],
        )