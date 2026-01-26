# =============================================================================
# METRICS - Survival metrics and evaluation functions
# =============================================================================

import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from sklearn.metrics import roc_auc_score, mean_squared_error
from sksurv.metrics import concordance_index_ipcw


@dataclass
class FoldMetrics:
    """Metrics for a single fold."""

    fold: int
    train_loss: float
    val_loss: float
    c_index: float = 0.5
    ipcw_c_index: float = 0.5
    auc: float = 0.5

    def __repr__(self):
        return (
            f"Fold {self.fold}: train_loss={self.train_loss:.4f}, "
            f"val_loss={self.val_loss:.4f}, ipcw_c={self.ipcw_c_index:.4f}"
        )


@dataclass
class CVResults:
    """Cross-validation results with statistics."""

    fold_metrics: List[FoldMetrics]
    mean_train_loss: float
    mean_val_loss: float
    mean_c_index: float
    mean_ipcw_c_index: float
    std_ipcw_c_index: float
    mean_auc: float = 0.5

    @classmethod
    def from_folds(cls, fold_metrics: List[FoldMetrics]) -> "CVResults":
        return cls(
            fold_metrics=fold_metrics,
            mean_train_loss=np.mean([f.train_loss for f in fold_metrics]).item(),
            mean_val_loss=np.mean([f.val_loss for f in fold_metrics]).item(),
            mean_c_index=np.mean([f.c_index for f in fold_metrics]).item(),
            mean_ipcw_c_index=np.mean([f.ipcw_c_index for f in fold_metrics]).item(),
            std_ipcw_c_index=np.std([f.ipcw_c_index for f in fold_metrics]).item(),
            mean_auc=np.mean([f.auc for f in fold_metrics]).item(),
        )

    def summary(self) -> str:
        lines = ["=" * 60, "CV RESULTS SUMMARY", "=" * 60]
        for m in self.fold_metrics:
            lines.append(str(m))
        lines.append("-" * 60)
        lines.append(f"Mean Train Loss: {self.mean_train_loss:.4f}")
        lines.append(f"Mean Val Loss:   {self.mean_val_loss:.4f}")
        lines.append(
            f"Mean IPCW C-idx: {self.mean_ipcw_c_index:.4f} ± {self.std_ipcw_c_index:.4f}"
        )
        if self.mean_auc > 0.5:
            lines.append(f"Mean AUC:        {self.mean_auc:.4f}")
        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean_train_loss": self.mean_train_loss,
            "mean_val_loss": self.mean_val_loss,
            "mean_c_index": self.mean_c_index,
            "mean_ipcw_c_index": self.mean_ipcw_c_index,
            "std_ipcw_c_index": self.std_ipcw_c_index,
            "mean_auc": self.mean_auc,
        }


def make_survival_array(times: np.ndarray, events: np.ndarray) -> np.ndarray:
    """Crée un structured array pour sksurv."""
    return np.array(
        [(bool(e), t) for e, t in zip(events, times)],
        dtype=[("event", bool), ("time", float)],
    )


def compute_ipcw_cindex(
    times_train: np.ndarray,
    events_train: np.ndarray,
    times_test: np.ndarray,
    events_test: np.ndarray,
    risk_scores: np.ndarray,
    tau: Optional[float] = None,
) -> float:
    """
    Calcule le C-index IPCW avec gestion d'erreurs.

    Returns:
        C-index IPCW ou 0.5 en cas d'erreur
    """
    try:
        surv_train = make_survival_array(times_train, events_train)
        surv_test = make_survival_array(times_test, events_test)
        if tau is None:
            tau = times_test.max()
        c_idx, _, _, _, _ = concordance_index_ipcw(
            surv_train, surv_test, risk_scores, tau=tau
        )
        return c_idx
    except Exception:
        return 0.5


def compute_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute MSE."""
    return mean_squared_error(y_true, y_pred)


def compute_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute AUC with error handling."""
    try:
        return roc_auc_score(y_true, y_pred)
    except Exception:
        return 0.5
