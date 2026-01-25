# =============================================================================
# OPTUNA UTILS - Avec IPCW C-index au lieu de concordance_index simple
# =============================================================================

import optuna
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, mean_squared_error
from sksurv.metrics import concordance_index_ipcw
from scipy.stats import norm
import numpy as np
import pandas as pd

optuna.logging.set_verbosity(optuna.logging.WARNING)

# --- Configuration ---
N_FOLDS = 5
RANDOM_STATE = 42


# --- Cross-Validation Factories ---
def get_stratified_kfold(n_splits=N_FOLDS):
    """Retourne un StratifiedKFold pour les problèmes de classification."""
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)


def get_kfold(n_splits=N_FOLDS):
    """Retourne un KFold standard pour les problèmes de régression."""
    return KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)


# --- Transformation de Y ---
def transform_y(y):
    """Transforme y via rank -> uniform -> inverse CDF normale."""
    n = len(y)
    ranks = (-y).argsort().argsort() + 1  # Rang basé sur -temps
    uniform_scores = ranks / (n + 1)
    return norm.ppf(uniform_scores)


def inverse_transform_pred(pred):
    """CDF normale pour obtenir des scores uniformes."""
    return norm.cdf(pred)


# --- Conversion vers format sksurv ---
def make_survival_array(time, event):
    """Crée un structured array pour sksurv: dtype=[('event', bool), ('time', float)]"""
    return np.array(
        [(bool(e), t) for e, t in zip(event, time)],
        dtype=[("event", bool), ("time", float)],
    )


# --- Métriques ---
def compute_bic(n, k, nll_or_mse, is_regression=True):
    """Calcule le BIC. Pour régression: log(MSE), pour classification: NLL."""
    if is_regression:
        if nll_or_mse <= 0:
            nll_or_mse = 1e-10
        return k * np.log(n) + n * np.log(nll_or_mse)
    else:
        return k * np.log(n) + 2 * nll_or_mse


def compute_aic(k, nll):
    """Calcule l'AIC pour classification."""
    return 2 * k + 2 * nll


def safe_concordance_index(y_true, pred, event):
    """C-index simple (lifelines) avec gestion d'erreurs."""
    from lifelines.utils import concordance_index

    try:
        return concordance_index(y_true, pred, event)
    except ZeroDivisionError:
        return 0.5


def safe_ipcw_cindex(
    y_train_time, y_train_event, y_test_time, y_test_event, pred, tau=None
):
    """
    IPCW C-index avec gestion d'erreurs.

    Args:
        y_train_time: Temps de survie sur train (pour estimer la censure)
        y_train_event: Événements sur train
        y_test_time: Temps de survie sur test
        y_test_event: Événements sur test
        pred: Prédictions de risque (plus élevé = plus de risque = survie plus courte)
        tau: Temps de troncature (optionnel, par défaut max du test)

    Returns:
        C-index IPCW
    """
    try:
        # Convertir en format sksurv
        survival_train = make_survival_array(y_train_time, y_train_event)
        survival_test = make_survival_array(y_test_time, y_test_event)

        # Tau = temps max observé dans le test si non spécifié
        if tau is None:
            tau = y_test_time.max()

        # IPCW C-index: estimate est la 2ème valeur retournée
        c_idx, _, _, _, _ = concordance_index_ipcw(
            survival_train, survival_test, pred, tau=tau
        )
        return c_idx
    except Exception as e:
        print(f"IPCW error: {e}")
        return 0.5  # Fallback


# --- Comptage des feuilles pour modèles à arbres ---
def count_total_leaves(model):
    """Compte le nombre total de feuilles d'un modèle à arbres."""
    model_type = type(model).__name__

    if "XGB" in model_type:
        try:
            df = model.get_booster().trees_to_dataframe()
            return (df["Feature"] == "Leaf").sum()
        except:
            n_trees = model.n_estimators
            max_depth = getattr(model, "max_depth", 6)
            return n_trees * (2**max_depth)

    elif "LGBM" in model_type:
        try:
            booster = model.booster_
            return sum(booster.num_leaves())
        except:
            return model.n_estimators * model.num_leaves

    elif "CatBoost" in model_type:
        try:
            tree_count = model.tree_count_
            depth = model.get_param("depth") or 6
            return tree_count * (2**depth)
        except:
            return model.get_param("iterations") * (2 ** model.get_param("depth"))

    elif "RandomForest" in model_type:
        try:
            return sum(tree.get_n_leaves() for tree in model.estimators_)
        except:
            return model.n_estimators * (2 ** getattr(model, "max_depth", 10))

    elif "AdaBoost" in model_type:
        try:
            return sum(est.get_n_leaves() for est in model.estimators_)
        except:
            return model.n_estimators * 2

    else:
        return 0


print("Optuna utils with IPCW C-index loaded successfully!")


# =============================================================================
# EXEMPLE D'UTILISATION DANS evaluate_model
# =============================================================================
"""
def evaluate_model(model, X, y_raw, event):
    kfold = get_kfold()
    c_indices, mses = [], []
    
    for train_idx, val_idx in kfold.split(X):
        X_tr, X_val = X.iloc[train_idx].values, X.iloc[val_idx].values
        y_tr_t = transform_y(y_raw[train_idx])
        
        try:
            model.fit(X_tr, y_tr_t)
            pred = model.predict(X_val)
            
            # Utiliser IPCW C-index au lieu de concordance_index
            # pred doit être un score de risque: plus élevé = survie plus courte
            risk_score = norm.cdf(pred)  # Transformer en [0,1], score de risque
            
            c_idx = safe_ipcw_cindex(
                y_train_time=y_raw[train_idx],
                y_train_event=event[train_idx],
                y_test_time=y_raw[val_idx],
                y_test_event=event[val_idx],
                pred=risk_score  # Plus élevé = plus de risque
            )
            c_indices.append(c_idx)
            mses.append(np.mean((pred - transform_y(y_raw[val_idx]))**2))
        except Exception as e:
            c_indices.append(0.5)
            mses.append(1.0)
            
    return np.mean(c_indices), np.mean(mses)
"""
