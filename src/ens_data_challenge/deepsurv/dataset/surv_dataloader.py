from torch.utils.data import DataLoader
import pytorch_lightning as pl
import pandas as pd
from typing import List

from .surv_dataset import CustomSurvDataset
from ens_data_challenge.preprocess import Preprocessor
from ens_data_challenge.feature_engineering import FeatureEngineer2
from ens_data_challenge.types import Columns, CytoColumns, MolecularColumns, CytoStructColumns

import torch
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import StandardScaler

def surv_collate_fn(batch):
    """
    Collate function qui pad les tenseurs de taille variable (molecular, cytogenetic)
    sur la dimension 0 (nombre de lignes par patient).
    """
    molecular_list = [item['molecular'] for item in batch]
    cytogenetic_list = [item['cytogenetic'] for item in batch]
    clinical_list = [item['clinical'] for item in batch]
    time_list = [item['target_time'] for item in batch]
    event_list = [item['target_event'] for item in batch]

    # Padding des matrices (dim 0)
    padded_molecular = pad_sequence(molecular_list, batch_first=True, padding_value=0.0)
    padded_cytogenetic = pad_sequence(cytogenetic_list, batch_first=True, padding_value=0.0)

    # Données cliniques : déjà 1D, donc on stack
    clinical = torch.stack(clinical_list)
    target_time = torch.stack(time_list)
    target_event = torch.stack(event_list)

    return {
        "molecular": padded_molecular,         # shape [B, max_n_mol, F_mol]
        "cytogenetic": padded_cytogenetic,     # shape [B, max_n_cyto, F_cyto]
        "clinical": clinical,                  # shape [B, F_clin]
        "target_time": target_time,            # shape [B]
        "target_event": target_event,          # shape [B]
        "lengths_molecular": torch.tensor([m.shape[0] for m in molecular_list]),
        "lengths_cytogenetic": torch.tensor([c.shape[0] for c in cytogenetic_list]),
    }


class SurvDataModule(pl.LightningDataModule):

    def __init__(
            self, 
            molecular_train_data: pd.DataFrame, 
            clinical_train_data: pd.DataFrame, 
            train_targets: pd.DataFrame, 
            clinical_val_data: pd.DataFrame,
            molecular_val_data: pd.DataFrame,
            val_targets: pd.DataFrame, 
            features: List[str],
            batch_size: int
            ):
        
        super().__init__()
        
        self.molecular_train_data = molecular_train_data
        self.clinical_train_data = clinical_train_data
        self.train_targets = train_targets
        self.molecular_val_data = molecular_val_data
        self.clinical_val_data = clinical_val_data
        self.val_targets = val_targets
        self.batch_size = batch_size
        self.features = features

    def setup(self, stage=None):

        pe = Preprocessor()

        cyto_struct_train, cyto_struct_val = pe.fit(
            clinical_data_train=self.clinical_train_data, 
            molecular_data_train=self.molecular_train_data,
            clinical_data_test=self.clinical_val_data,
            molecular_data_test=self.molecular_val_data,
        )

        clinical_train_clean, molecular_train_clean, cytogenetic_train_clean, train_targets_clean = pe.transform(
            self.clinical_train_data, self.molecular_train_data, cyto_struct_train, self.train_targets
        )

        clinical_val_clean, molecular_val_clean, cytogenetic_val_clean, val_targets_clean = pe.transform(
            self.clinical_val_data, self.molecular_val_data, cyto_struct_val, self.val_targets
        )

        fe = FeatureEngineer2(
            clinical_data_train=clinical_train_clean, 
            molecular_data_train=molecular_train_clean,
            cytogenetic_data_train=cytogenetic_train_clean,
            clinical_data_test=clinical_val_clean,
            molecular_data_test=molecular_val_clean,
            cytogenetic_data_test=cytogenetic_val_clean,
            target_data_train=train_targets_clean,
        )
        
        fe.as_type(
            type_dict={
                CytoColumns.IS_NORMAL.value: 'int32',
                # CytoColumns.HAS_TP53_DELETION.value: 'int32',
                # CytoColumns.HAS_COMPLEX_CHR3.value: 'int32',
                # CytoColumns.HAS_LARGE_DELETION.value: 'int32',
            },
            data_type='clinical'
        )

        # fe.add_mol_encoding(
        #     col='GENE',
        #     method='vaf_score',
        #     apply_effect_weighting=False
        # )

        fe.add_mol_encoding(
            col='PATHWAY',
            method='constant',
            apply_effect_weighting=False
        )

        # fe.add_mol_encoding(
        #     col='EFFECT',
        #     method='constant',
        #     apply_effect_weighting=False
        # )

        fe.encode_risk(
            categorical_cols=[
                CytoColumns.MDS_IPSS_R_CYTO_RISK.value,
                CytoColumns.AML_ELN_2022_CYTO_RISK.value,
                CytoColumns.CLL_CYTO_RISK.value,
                CytoColumns.MM_RISS_CYTO_RISK.value,
            ],
            data_type='clinical'
        )

        fe.one_hot_encode(
            categorical_cols=[
                MolecularColumns.EFFECT.value,
                MolecularColumns.GENE.value,
                MolecularColumns.REF.value,
                MolecularColumns.ALT.value,
                MolecularColumns.CHR.value
            ],
            data_type='molecular'
        )

        fe.one_hot_encode(
            categorical_cols=[
                CytoStructColumns.MUTATION_TYPE.value,
                CytoStructColumns.ARM.value,
                CytoStructColumns.SEX_CHROMOSOMES.value,
                CytoStructColumns.END_ARM.value,
                CytoStructColumns.START_ARM.value,
            ],
            data_type='cytogenetic'
        )

        clinical_train_processed, clinical_val_processed = fe.get_clinical_data()
        molecular_train_processed, molecular_val_processed = fe.get_molecular_data()
        cytogenetic_train_processed, cytogenetic_val_processed = fe.get_cytogenetic_data()


    
        
        # Trier par ID pour aligner les données
        clinical_train_processed = clinical_train_processed.sort_values('ID').reset_index(drop=True)
        molecular_train_processed = molecular_train_processed.sort_values('ID').reset_index(drop=True)
        cytogenetic_train_processed = cytogenetic_train_processed.sort_values('ID').reset_index(drop=True)
        train_targets_clean = train_targets_clean.sort_values('ID').reset_index(drop=True)

        clinical_val_processed = clinical_val_processed.sort_values('ID').reset_index(drop=True)
        molecular_val_processed = molecular_val_processed.sort_values('ID').reset_index(drop=True)
        cytogenetic_val_processed = cytogenetic_val_processed.sort_values('ID').reset_index(drop=True)
        val_targets_clean = val_targets_clean.sort_values('ID').reset_index(drop=True)

        molecular_train_processed.drop(columns=["START", "END", "DEPTH", ], inplace=True)
        molecular_val_processed.drop(columns=["START", "END", "DEPTH", ], inplace=True)

        print(f"Final clinical features: {clinical_train_processed.columns.tolist()}")
        print(f"Final molecular features: {molecular_train_processed.columns.tolist()}")
        print(f"Final cytogenetic features: {cytogenetic_train_processed.columns.tolist()}")

        # Standard scaling
        clinical_scaler = StandardScaler()
        clinical_train_processed.iloc[:, :-1] = clinical_scaler.fit_transform(clinical_train_processed.iloc[:, :-1])  # skip ID
        clinical_val_processed.iloc[:, :-1] = clinical_scaler.transform(clinical_val_processed.iloc[:, :-1])

        mol_scaler = StandardScaler()
        molecular_train_processed.iloc[:, :-1] = mol_scaler.fit_transform(molecular_train_processed.iloc[:, :-1])  # skip ID
        molecular_val_processed.iloc[:, :-1] = mol_scaler.transform(molecular_val_processed.iloc[:, :-1])

        cyto_scaler = StandardScaler()
        cytogenetic_train_processed.iloc[:, :-1] = cyto_scaler.fit_transform(cytogenetic_train_processed.iloc[:, :-1])  # skip ID
        cytogenetic_val_processed.iloc[:, :-1] = cyto_scaler.transform(cytogenetic_val_processed.iloc[:, :-1])

        if clinical_train_processed.isna().any().any():
            print("NaN detected in clinical training data")
        
        clinical_train_processed.fillna(0.0, inplace=True)
        clinical_val_processed.fillna(0.0, inplace=True)
            
        self.train_dataset = CustomSurvDataset(
            molecularData=molecular_train_processed, 
            clinicalData=clinical_train_processed, 
            cytogeneticData=cytogenetic_train_processed,
            targets=train_targets_clean
        )

        self.val_dataset = CustomSurvDataset(
            molecularData=molecular_val_processed,
            clinicalData=clinical_val_processed,
            cytogeneticData=cytogenetic_val_processed,
            targets=val_targets_clean
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=surv_collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=surv_collate_fn)