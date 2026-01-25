from typing import Protocol
import pandas as pd

class EventClassifierTrainer:
    
    def  __init__(self) -> None:
        self.time_col = "OS_YEARS"
        self.event_col = "OS_STATUS"

    
    def train(self, data: pd.DataFrame, targets: pd.DataFrame, model: "ModelWrapper"):
        """Takes a preprocess, engineered DataFrame and a model, and trains the model.

        Args:
            data: DataFrame with features
            targets: DataFrame with target columns
            model: ModelWrapper instance with fit method
        Returns: None
        """
        
        model.fit(X=data, y=targets[self.event_col])