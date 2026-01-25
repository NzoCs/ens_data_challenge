# =============================================================================
# Kaplan-Meier Weights Approximation (Pandas version)
# =============================================================================
# Fonctions pour calculer les poids IPCW (Inverse Probability of Censoring Weighting)
# basés sur l'estimateur de Kaplan-Meier de la censure
# =============================================================================

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Union


def compute_kaplan_meier_weights(
    df: pd.DataFrame,
    time_col: str = "OS_YEARS",
    event_col: str = "OS_STATUS",
    tau: Optional[float] = None,
) -> pd.DataFrame:
    """
    Calcule les poids IPCW basés sur l'estimateur Kaplan-Meier de la censure.

    Pour les données censurées, on pondère les observations par l'inverse de
    la probabilité d'être encore observable à ce temps (pas censuré).

    Args:
        df: DataFrame avec colonnes OS_YEARS et OS_STATUS
        time_col: Nom de la colonne temps
        event_col: Nom de la colonne événement (1=événement, 0=censuré)
        tau: Temps de troncature optionnel (exclut les temps > tau)

    Returns:
        DataFrame avec colonnes ajoutées:
            - 'G_t': Probabilité de survie de la censure G(t)
            - 'ipcw_weight': Poids IPCW (1/G(t-) si événement, 0 si censuré)

    Notes:
        - Les événements (event=1) reçoivent un poids = 1/G(t-)
        - Les censures (event=0) reçoivent un poids = 0 (non informatif)
        - G(t) est la probabilité de NE PAS être censuré avant t
    """
    result = df.copy()

    # Trier par temps
    sorted_df = df.sort_values(time_col).reset_index(drop=True)

    # Pour KM de la censure: censure = événement d'intérêt
    sorted_df["_censoring"] = 1 - sorted_df[event_col]

    # Calculer at_risk et n_censored par temps unique
    time_stats = (
        sorted_df.groupby(time_col)
        .agg(n_censored=("_censoring", "sum"), n_total=("_censoring", "count"))
        .reset_index()
    )

    # Calculer at_risk cumulatif (de la fin vers le début)
    n = len(sorted_df)
    time_stats["at_risk"] = n - time_stats["n_total"].cumsum().shift(1, fill_value=0)

    # Calculer G(t) avec produit cumulatif Kaplan-Meier
    time_stats["hazard"] = time_stats["n_censored"] / time_stats["at_risk"].clip(
        lower=1
    )
    time_stats["survival_factor"] = 1 - time_stats["hazard"]
    time_stats["G_t"] = time_stats["survival_factor"].cumprod()

    # Créer un mapping temps -> G(t-)
    time_stats["G_t_minus"] = time_stats["G_t"].shift(1, fill_value=1.0)

    # Merger avec les données originales
    km_lookup = time_stats[[time_col, "G_t", "G_t_minus"]].copy()
    result = result.merge(km_lookup, on=time_col, how="left")

    # Calculer les poids IPCW
    result["ipcw_weight"] = 0.0

    # Événements: poids = 1 / G(t-)
    event_mask = result[event_col] == 1
    result.loc[event_mask, "ipcw_weight"] = 1.0 / result.loc[
        event_mask, "G_t_minus"
    ].clip(lower=1e-10) # type: ignore

    # Appliquer tau si spécifié
    if tau is not None:
        result.loc[result[time_col] > tau, "ipcw_weight"] = 0.0

    return result


def get_ipcw_weights(
    times: Union[pd.Series, np.ndarray],
    events: Union[pd.Series, np.ndarray],
    tau: Optional[float] = None,
) -> pd.Series:
    """
    Version simplifiée pour obtenir directement les poids IPCW.

    Args:
        times: Series ou array des temps
        events: Series ou array des événements (1=événement, 0=censuré)
        tau: Temps de troncature optionnel

    Returns:
        pd.Series: Poids IPCW
    """
    df = pd.DataFrame({"time": times, "event": events})
    result = compute_kaplan_meier_weights(df, "time", "event", tau)
    return result["ipcw_weight"]


def kaplan_meier_curve(
    df: pd.DataFrame,
    time_col: str = "OS_YEARS",
    event_col: str = "OS_STATUS",
    for_censoring: bool = False,
) -> pd.DataFrame:
    """
    Calcule la courbe de survie Kaplan-Meier.

    Args:
        df: DataFrame avec colonnes time et event
        time_col: Nom de la colonne temps
        event_col: Nom de la colonne événement
        for_censoring: Si True, calcule G(t) (survie de la censure)
                       Si False, calcule S(t) (survie des événements)

    Returns:
        DataFrame avec colonnes: time, at_risk, n_events, S_t (ou G_t)
    """
    sorted_df = df.sort_values(time_col).copy()

    if for_censoring:
        sorted_df["_event"] = 1 - sorted_df[event_col]
    else:
        sorted_df["_event"] = sorted_df[event_col]

    # Stats par temps unique
    time_stats = (
        sorted_df.groupby(time_col)
        .agg(n_events=("_event", "sum"), n_total=("_event", "count"))
        .reset_index()
    )

    n = len(sorted_df)
    time_stats["at_risk"] = n - time_stats["n_total"].cumsum().shift(1, fill_value=0)

    # Kaplan-Meier
    time_stats["hazard"] = time_stats["n_events"] / time_stats["at_risk"].clip(lower=1)
    time_stats["survival"] = (1 - time_stats["hazard"]).cumprod()

    col_name = "G_t" if for_censoring else "S_t"
    time_stats = time_stats.rename(columns={"survival": col_name})

    return time_stats[[time_col, "at_risk", "n_events", col_name]]


def compute_weighted_loss(
    df: pd.DataFrame,
    pred_col: str,
    true_col: str,
    time_col: str = "OS_YEARS",
    event_col: str = "OS_STATUS",
    loss_fn: str = "mse",
    tau: Optional[float] = None,
) -> float:
    """
    Calcule une loss pondérée par les poids IPCW.

    Args:
        df: DataFrame avec les données
        pred_col: Nom de la colonne prédictions
        true_col: Nom de la colonne vraies valeurs
        time_col: Nom de la colonne temps
        event_col: Nom de la colonne événement
        loss_fn: 'mse', 'mae', ou 'huber'
        tau: Temps de troncature optionnel

    Returns:
        Loss pondérée IPCW normalisée
    """
    result = compute_kaplan_meier_weights(df, time_col, event_col, tau)
    weights = result["ipcw_weight"]

    pred = result[pred_col]
    true = result[true_col]

    if loss_fn == "mse":
        losses = (pred - true) ** 2
    elif loss_fn == "mae":
        losses = (pred - true).abs()
    elif loss_fn == "huber":
        delta = 1.0
        abs_diff = (pred - true).abs()
        losses = pd.Series(
            np.where(
                abs_diff <= delta,
                0.5 * (pred - true) ** 2,
                delta * (abs_diff - 0.5 * delta),
            )
        )
    else:
        raise ValueError(f"Unknown loss function: {loss_fn}")

    # Pondérer et normaliser
    weighted_sum = (weights * losses).sum()
    weight_sum = weights.sum()

    if weight_sum > 0:
        return weighted_sum / weight_sum
    else:
        return 0.0


# --- Exemple d'utilisation ---
if __name__ == "__main__":
    # Test avec données synthétiques
    np.random.seed(42)
    n = 100

    df = pd.DataFrame(
        {
            "patient_id": range(n),
            "time": np.random.exponential(5, n),
            "event": np.random.binomial(1, 0.7, n),
        }
    )

    # Calcul des poids
    result = compute_kaplan_meier_weights(df, "time", "event")

    print("Exemple de poids IPCW:")
    print(result[["time", "event", "G_t", "ipcw_weight"]].head(10))

    print(f"\nStats poids:")
    print(
        f"  Min (non-zero): {result.loc[result['ipcw_weight'] > 0, 'ipcw_weight'].min():.3f}"
    )
    print(f"  Max: {result['ipcw_weight'].max():.3f}")
    print(
        f"  Mean (non-zero): {result.loc[result['ipcw_weight'] > 0, 'ipcw_weight'].mean():.3f}"
    )

    # Courbe KM
    km = kaplan_meier_curve(df, "time", "event", for_censoring=True)
    print(f"\nCourbe G(t) censure:")
    print(km.head(10))
