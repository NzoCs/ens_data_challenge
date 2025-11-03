import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split

from ens_data_challenge.deepsurv.model.deepsurv import DeepSurv
from ens_data_challenge.deepsurv.model.MLP import DefaultMLP
from ens_data_challenge.deepsurv.dataset.surv_dataloader import SurvDataModule
from ens_data_challenge.globals import (
    TRAIN_CLINICAL_DATA_PATH,
    TRAIN_MOLECULAR_DATA_PATH,
    TRAIN_TARGET_PATH,
    TEST_CLINICAL_DATA_PATH,
    TEST_MOLECULAR_DATA_PATH
)

def main():
    # Load data
    clinical_train = pd.read_csv(TRAIN_CLINICAL_DATA_PATH)
    molecular_train = pd.read_csv(TRAIN_MOLECULAR_DATA_PATH)
    target_train = pd.read_csv(TRAIN_TARGET_PATH)

    # For simplicity, split train into train and val
    # In real scenario, use separate val set or cross-validation
    train_ids, val_ids = train_test_split(clinical_train['ID'].unique(), test_size=0.2, random_state=42)

    clinical_train_data = clinical_train[clinical_train['ID'].isin(train_ids)]
    molecular_train_data = molecular_train[molecular_train['ID'].isin(train_ids)]
    train_targets = target_train[target_train['ID'].isin(train_ids)]

    clinical_val_data = clinical_train[clinical_train['ID'].isin(val_ids)]
    molecular_val_data = molecular_train[molecular_train['ID'].isin(val_ids)]
    val_targets = target_train[target_train['ID'].isin(val_ids)]

    # Features to generate
    features = ["Nmut"]  # Add more as needed

    # Data module
    dm = SurvDataModule(
        molecular_train_data=molecular_train_data,
        clinical_train_data=clinical_train_data,
        train_targets=train_targets,
        clinical_val_data=clinical_val_data,
        molecular_val_data=molecular_val_data,
        val_targets=val_targets,
        features=features,
        batch_size=32
    )

    # Model
    model = DeepSurv(
        model=DefaultMLP(hidden_layers=[16, 8, 4]),
        mol_embed_dim=4,
        mol_hidden_dim=4,
        cyto_embed_dim=4,
        cyto_hidden_dim=4,
        loss_type="smooth_cindex",
        lr=1e-3,
        weight_decay=1e-5
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_cindex_ipcw',
        mode='max',
        save_top_k=1,
        filename='deepsurv-{epoch:02d}-{val_cindex_ipcw:.2f}'
    )

    early_stop_callback = EarlyStopping(
        monitor='val_cindex_ipcw',
        mode='max',
        patience=1000,
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=100,
        callbacks=[checkpoint_callback, early_stop_callback],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1
    )

    # Train
    trainer.fit(model, dm)

    # # Test on val (for now)
    # trainer.test(model, dm)

if __name__ == "__main__":
    main()