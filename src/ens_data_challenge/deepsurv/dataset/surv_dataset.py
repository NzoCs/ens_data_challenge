import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sksurv.util import Surv

from typing import Dict


class CustomSurvDataset(Dataset):

    def __init__(
            self, 
            molecularData: pd.DataFrame, 
            clinicalData: pd.DataFrame, 
            cytogeneticData: pd.DataFrame, 
            targets: pd.DataFrame
            ):

        self.molecularData = molecularData.groupby('ID')
        self.cytogeneticData = cytogeneticData.groupby('ID')
        self.clinicalData = clinicalData
        targets = targets.set_index('ID')
        # Liste des IDs (pour itérer facilement par idx)
        self.ids = list(self.clinicalData['ID'])
        self.targets_time = targets["OS_YEARS"]
        self.targets_event = targets["OS_STATUS"]

        # Calculate number of features
        self.num_molecular_features = molecularData.shape[1] - 1
        self.num_cytogenetic_features = cytogeneticData.shape[1] - 1
        self.num_clinical_features = clinicalData.shape[1] - 1


    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        patient_id = self.ids[idx]

        # Toutes les lignes du patient
        try:
            molecular = self.molecularData.get_group(patient_id).drop(columns='ID').values
        except KeyError:
            molecular = np.zeros((0, self.num_molecular_features))
        
        try:
            cytogenetic = self.cytogeneticData.get_group(patient_id).drop(columns='ID').values
        except KeyError:
            cytogenetic = np.zeros((0, self.num_cytogenetic_features))

        # Données cliniques (une seule ligne par patient)
        clinical = self.clinicalData[self.clinicalData['ID'] == patient_id].drop(columns='ID').values.squeeze()

        targets_time = self.targets_time.loc[patient_id]
        targets_event = self.targets_event.loc[patient_id]

        return {
            'molecular': torch.tensor(molecular, dtype=torch.float32).sum(dim=0, keepdim=True),
            'cytogenetic': torch.tensor(cytogenetic, dtype=torch.float32).sum(dim=0, keepdim=True),  # moyenne si plusieurs lignes
            'clinical': torch.tensor(clinical, dtype=torch.float32),
            'target_time': torch.tensor(targets_time, dtype=torch.float32),
            'target_event': torch.tensor(targets_event, dtype=torch.float32),
        }