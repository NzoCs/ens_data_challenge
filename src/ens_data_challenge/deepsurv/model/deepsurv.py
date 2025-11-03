from sksurv.metrics import concordance_index_ipcw
import numpy as np
from sksurv.util import Surv


import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
from typing import Dict, Literal, Tuple

from ens_data_challenge.deepsurv.model.set_embedding import SetAttention

class DeepSurv(pl.LightningModule):
    def __init__(
            self, 
            model: nn.Module,
            mol_embed_dim: int,
            mol_hidden_dim: int,
            cyto_embed_dim: int,
            cyto_hidden_dim: int,
            loss_type: Literal["cox", "smooth_cindex"] = "cox",
            smooth_margin: float = 0.1,
            regularization: float = 0.0,
            l2_ratio: float = 1.0,
            lr: float = 1e-3, 
            weight_decay: float = 1e-5
            ):
        
        super().__init__()

        self.save_hyperparameters(ignore=["model"])
        self.model = model

        self.mol_embedding = SetAttention(
            hidden_dim=mol_hidden_dim, 
            output_dim=mol_embed_dim
            )
        
        self.cyto_embedding = SetAttention(
            hidden_dim=cyto_hidden_dim, 
            output_dim=cyto_embed_dim,
        )

        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_type = loss_type
        self.smooth_margin = smooth_margin
        self.regularization = regularization
        self.l2_ratio = l2_ratio

    def forward(self, x_molecular: torch.Tensor, x_cytogenetic: torch.Tensor, x_clinical: torch.Tensor) -> torch.Tensor:
        embedded_mol = self.mol_embedding(x_molecular)
        embedded_cyto = self.cyto_embedding(x_cytogenetic)
        x = torch.cat([embedded_mol, embedded_cyto, x_clinical], dim=1)
        return self.model(x)

    def smooth_cindex_loss(self, risk_pred: torch.Tensor, time: torch.Tensor, event: torch.Tensor, smooth_margin: float) -> torch.Tensor:
        """
        Vectorized differentiable approximation of the concordance index (Haider et al., Bioinformatics, 2020).
        
        Args:
            risk_pred (torch.Tensor): Predicted risk scores (higher = higher risk).
            time (torch.Tensor): Survival times.
            event (torch.Tensor): Event indicators (1 if event occurred, 0 if censored).
            smooth_margin (float): Temperature parameter for the sigmoid (default=0.1).

        Returns:
            torch.Tensor: Scalar tensor representing the smooth C-index loss.
        """
        # Ensure tensors are of correct shape
        risk_pred = risk_pred.view(-1)
        time = time.view(-1)
        event = event.view(-1)

        # Create comparison masks (only pairs i,j where time[i] < time[j] and event[i] == 1)
        mask = (time.unsqueeze(0) < time.unsqueeze(1)) & (event.unsqueeze(0) == 1)

        # Compute pairwise differences
        diff = risk_pred.unsqueeze(1) - risk_pred.unsqueeze(0)  # shape (n, n)

        # Apply sigmoid smoothing
        loss = torch.sigmoid(diff / smooth_margin)

        # Keep only valid pairs
        if mask.sum() == 0:
            return torch.tensor(0.0, device=risk_pred.device)

        # Average over comparable pairs
        loss = loss[mask].mean()
        return loss

    def cox_ph_loss(self, risk_pred: torch.Tensor, time: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
        order = torch.argsort(time, descending=True)
        risk_pred = risk_pred[order]
        event = event[order]
        log_cum_risk = torch.logcumsumexp(risk_pred, dim=0)
        diff = risk_pred - log_cum_risk
        loss = -torch.sum(diff * event) / torch.sum(event)
        return loss

    # ============================
    #       Training step
    # ============================
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> STEP_OUTPUT:
        x_mol = batch["molecular"]
        x_cyto = batch["cytogenetic"]
        x_clin = batch["clinical"]
        time = batch["target_time"]
        event = batch["target_event"]

        risk_pred = self.forward(x_mol, x_cyto, x_clin).squeeze()

        if self.loss_type == "cox":
            loss = self.cox_ph_loss(risk_pred, time, event)
        elif self.loss_type == "smooth_cindex":
            loss = self.smooth_cindex_loss(risk_pred, time, event, self.smooth_margin)
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        self.log("train_loss", loss, prog_bar=True, on_epoch=True)

        return 

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> STEP_OUTPUT:

        x_mol = batch["molecular"]
        x_cyto = batch["cytogenetic"]
        x_clin = batch["clinical"]
        y_time = batch["target_time"]
        y_event = batch["target_event"]

        risk_pred = self.forward(x_mol, x_cyto, x_clin).squeeze()

        # Compute loss for backprop reference
        loss = self.cox_ph_loss(risk_pred, y_time, y_event)

        # Compute concordance index (non-différentiable)
        time_np = y_time.detach().cpu().numpy()
        event_np = y_event.detach().cpu().numpy().astype(bool)
        risk_np = risk_pred.detach().cpu().numpy()

        # Pour concordance_index_ipcw, on a besoin d’un objet structuré
        y_struct = Surv.from_arrays(event_np, time_np)
        cindex = concordance_index_ipcw(y_struct, y_struct, -risk_np, tau=7)[0]  # - car plus haut risque = moins de survie

        self.log("val_loss", loss , prog_bar=True)
        self.log("val_cindex_ipcw", cindex * 100, prog_bar=True)
        return {"val_loss": loss, "val_cindex_ipcw": cindex}

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> STEP_OUTPUT:
        x_mol = batch["molecular"]
        x_cyto = batch["cytogenetic"]
        x_clin = batch["clinical"]
        y_time = batch["target_time"]
        y_event = batch["target_event"]

        risk_pred = self.forward(x_mol, x_cyto, x_clin).squeeze()

        y_struct = Surv.from_arrays(y_event.detach().cpu().numpy().astype(bool),
                                    y_time.detach().cpu().numpy())
        
        risk_items = -risk_pred.detach().cpu().numpy()  # - car plus haut risque = moins de survie

        cindex = concordance_index_ipcw(y_struct, y_struct, risk_items, tau=7)[0]

        self.log("test_cindex_ipcw", cindex * 100, prog_bar=True)

        return {"test_cindex_ipcw": cindex}

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
