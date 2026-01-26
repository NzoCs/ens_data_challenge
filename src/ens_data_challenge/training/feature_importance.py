# =============================================================================
# FEATURE IMPORTANCE - Pretraining analysis and feature selection
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import optuna
from sklearn.model_selection import StratifiedKFold, KFold
import warnings

from .model_factories import get_classifier_factory, get_regressor_factory
from .metrics import compute_auc, compute_mse

warnings.filterwarnings("ignore")


@dataclass
class FeatureImportanceResult:
    """Results of feature importance analysis."""

    feature_names: List[str]
    tree_importances: Dict[
        str, pd.Series
    ]  # model_name -> pd.Series (index=feature_names)
    linear_importances: Dict[str, pd.Series]
    mean_tree_importance: np.ndarray  # Valeurs uniquement (sans random)
    mean_linear_importance: np.ndarray
    random_feature_name: str
    random_tree_importance: float
    random_linear_importance: float
    # Sélections séparées
    tree_selected_features: List[str]  # Features significatives pour tree-based
    linear_selected_features: List[str]  # Features significatives pour linear
    selected_features: List[str]  # Union (tree OR linear)
    dropped_features: List[str]

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "FEATURE IMPORTANCE SUMMARY",
            "=" * 60,
            f"Total features: {len(self.feature_names)}",
            f"Random feature importance (tree): {self.random_tree_importance:.6f}",
            f"Random feature importance (linear): {self.random_linear_importance:.6f}",
            f"Tree-based selected: {len(self.tree_selected_features)}",
            f"Linear selected: {len(self.linear_selected_features)}",
            f"Union (tree OR linear): {len(self.selected_features)}",
            f"Dropped features: {len(self.dropped_features)}",
            "=" * 60,
        ]
        return "\n".join(lines)


def add_random_feature(
    X: pd.DataFrame, random_state: int = 42
) -> Tuple[pd.DataFrame, str]:
    """
    Ajoute une feature aléatoire pour servir de baseline.

    Returns:
        DataFrame avec random feature, nom de la random feature
    """
    np.random.seed(random_state)
    X = X.copy()
    random_col = "__RANDOM_FEATURE__"
    X[random_col] = np.random.randn(len(X))
    return X, random_col


def _optimize_model_params(
    X: pd.DataFrame,
    y: np.ndarray,
    model_name: str,
    is_classifier: bool = True,
    n_trials: int = 15,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Optimise les hyperparamètres d'un modèle avec Optuna (mini-round).
    """
    if is_classifier:
        factory = get_classifier_factory(model_name, random_state)
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
    else:
        factory = get_regressor_factory(model_name, random_state)
        kf = KFold(n_splits=3, shuffle=True, random_state=random_state)

    def objective(trial):
        scores = []
        # Pour linear models, on veut éviter d'être trop lent donc on sous-sample si besoin
        # Mais ici n_splits=3 c'est rapide.

        splitter = skf if is_classifier else kf
        for train_idx, val_idx in splitter.split(X, y):
            model = factory(trial)
            model.fit(X.iloc[train_idx], y[train_idx])

            if is_classifier:
                pred = model.predict_proba(X.iloc[val_idx])[:, 1]
                score = compute_auc(y[val_idx], pred)
                # Maximize AUC -> Optuna minimize -AUC
                scores.append(-score)
            else:
                pred = model.predict(X.iloc[val_idx])
                score = compute_mse(y[val_idx], pred)
                # Minimize MSE
                scores.append(score)

        return np.mean(scores)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def compute_tree_feature_importance(
    X: pd.DataFrame,
    y: np.ndarray,
    is_classifier: bool = True,
    random_state: int = 42,
) -> Dict[str, pd.Series]:
    """
    Calcule la feature importance avec des modèles à arbres.

    Returns:
        Dict[model_name -> pd.Series with feature names as index]
    """
    feature_names = list(X.columns)
    results = {}

    if is_classifier:
        model_names = ["RF", "XGB", "LGBM"]
    else:
        model_names = ["RF", "XGB", "LGBM"]

    for name in model_names:
        print(f"  Optimizing {name}...")
        best_params = _optimize_model_params(
            X, y, name, is_classifier, n_trials=15, random_state=random_state
        )

        # Instantiate with best params
        trial = optuna.trial.FixedTrial(best_params)
        if is_classifier:
            model = get_classifier_factory(name, random_state)(trial)
        else:
            model = get_regressor_factory(name, random_state)(trial)

        model.fit(X, y)
        importances = model.feature_importances_
        # Normaliser en [0, 1] pour comparabilité entre modèles
        imp_min, imp_max = importances.min(), importances.max()
        if imp_max > imp_min:
            importances = (importances - imp_min) / (imp_max - imp_min)
        # Créer une Series avec les noms de features comme index
        results[name] = pd.Series(importances, index=feature_names)

    return results


def compute_linear_feature_importance(
    X: pd.DataFrame, y: np.ndarray, is_classifier: bool = True, random_state: int = 42
) -> Dict[str, pd.Series]:
    """
    Calcule la feature importance avec des modèles linéaires (coefficients absolus).

    Returns:
        Dict[model_name -> pd.Series with feature names as index]
    """

    feature_names = list(X.columns)

    feature_names = list(X.columns)

    # Factories now use pipelines with StandardScaler, so we pass X directly.
    if is_classifier:
        print("  Optimizing LR...")
        name = "LR"
        # Factories handle standardization, so we don't scale X here
        best_params = _optimize_model_params(
            X, y, name, is_classifier, n_trials=10, random_state=random_state
        )
        trial = optuna.trial.FixedTrial(best_params)
        model = get_classifier_factory(name, random_state)(trial)
        model.fit(X, y)

        # Model is a pipeline: [StandardScaler, LogisticRegression]
        # Access estimator
        if hasattr(model, "steps"):
            estimator = model.steps[-1][1]
        else:
            estimator = model

        importances = np.abs(estimator.coef_[0])
    else:
        print("  Optimizing Ridge...")
        name = "Ridge"
        best_params = _optimize_model_params(
            X, y, name, is_classifier, n_trials=10, random_state=random_state
        )
        trial = optuna.trial.FixedTrial(best_params)
        model = get_regressor_factory(name, random_state)(trial)
        model.fit(X, y)

        # Model is a pipeline
        if hasattr(model, "steps"):
            estimator = model.steps[-1][1]
        else:
            estimator = model

        importances = np.abs(estimator.coef_)

    # Normaliser en [0, 1] pour cohérence avec les arbres
    imp_min, imp_max = importances.min(), importances.max()
    if imp_max > imp_min:
        importances = (importances - imp_min) / (imp_max - imp_min)

    # Créer une Series avec les noms de features comme index
    return {"Linear": pd.Series(importances, index=feature_names)}


def analyze_feature_importance(
    X: pd.DataFrame,
    y: np.ndarray,
    is_classifier: bool = True,
    threshold_method: str = "random",  # "random" or "quantile"
    quantile_threshold: float = 0.1,
    random_state: int = 42,
    verbose: bool = True,
) -> FeatureImportanceResult:
    """
    Analyse complète de la feature importance.

    Args:
        X: Features DataFrame
        y: Target array
        is_classifier: True for classification, False for regression
        threshold_method: "random" (drop features below random feature) or "quantile"
        quantile_threshold: Quantile threshold if method is "quantile"

    Returns:
        FeatureImportanceResult with selected and dropped features
    """
    # Add random feature
    X_with_random, random_col = add_random_feature(X, random_state)
    feature_names = list(X_with_random.columns)

    if verbose:
        print("Computing tree-based feature importance...")
    tree_importances = compute_tree_feature_importance(
        X_with_random, y, is_classifier, random_state=random_state
    )

    if verbose:
        print("Computing linear feature importance...")
    linear_importances = compute_linear_feature_importance(
        X_with_random, y, is_classifier, random_state=random_state
    )

    # Compute means (using DataFrame for proper alignment)
    tree_df = pd.DataFrame(tree_importances)
    linear_df = pd.DataFrame(linear_importances)

    mean_tree = tree_df.mean(axis=1)  # Series with feature names as index
    mean_linear = linear_df.mean(axis=1)  # Series with feature names as index

    # Get random feature importance (using proper indexing by name)
    random_tree_imp = mean_tree[random_col]
    random_linear_imp = mean_linear[random_col]

    if verbose:
        print(
            f"Random feature importance - Tree: {random_tree_imp:.4f}, Linear: {random_linear_imp:.4f}"
        )

    # Select features (comparing by name, not index)
    if threshold_method == "random":
        tree_mask = mean_tree > random_tree_imp
        linear_mask = mean_linear > random_linear_imp
        keep_mask = tree_mask | linear_mask
    else:
        tree_threshold = mean_tree.quantile(quantile_threshold)
        linear_threshold = mean_linear.quantile(quantile_threshold)
        tree_mask = mean_tree > tree_threshold
        linear_mask = mean_linear > linear_threshold
        keep_mask = tree_mask | linear_mask

    # Remove random feature from results
    real_features = [f for f in feature_names if f != random_col]

    tree_selected = [f for f in real_features if tree_mask[f]]
    linear_selected = [f for f in real_features if linear_mask[f]]
    selected_features = [f for f in real_features if keep_mask[f]]
    dropped_features = [f for f in real_features if not keep_mask[f]]

    # Filter Series to exclude random feature
    mean_tree_filtered = mean_tree.drop(random_col)
    mean_linear_filtered = mean_linear.drop(random_col)
    tree_importances_filtered = {
        k: v.drop(random_col) for k, v in tree_importances.items()
    }
    linear_importances_filtered = {
        k: v.drop(random_col) for k, v in linear_importances.items()
    }

    result = FeatureImportanceResult(
        feature_names=real_features,
        tree_importances=tree_importances_filtered,
        linear_importances=linear_importances_filtered,
        mean_tree_importance=mean_tree_filtered.values,
        mean_linear_importance=mean_linear_filtered.values,
        random_feature_name=random_col,
        random_tree_importance=random_tree_imp,
        random_linear_importance=random_linear_imp,
        tree_selected_features=tree_selected,
        linear_selected_features=linear_selected,
        selected_features=selected_features,
        dropped_features=dropped_features,
    )

    if verbose:
        print(result.summary())

    return result


def plot_feature_importance(
    result: FeatureImportanceResult,
    top_n: int = 30,
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot feature importance comparison.
    Colors are based on whether the feature is selected for THAT specific model type.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Tree importance - Top N (coloring based on tree_selected_features)
    ax = axes[0, 0]
    sorted_idx = np.argsort(result.mean_tree_importance)[::-1][:top_n]
    colors = [
        "green" if result.feature_names[i] in result.tree_selected_features else "red"
        for i in sorted_idx
    ]
    ax.barh(
        range(len(sorted_idx)), result.mean_tree_importance[sorted_idx], color=colors
    )
    ax.axvline(
        result.random_tree_importance,
        color="blue",
        linestyle="--",
        label="Random Feature",
    )
    ax.set_yticks(range(len(sorted_idx)))
    ax.set_yticklabels([result.feature_names[i][:30] for i in sorted_idx], fontsize=8)
    ax.set_xlabel("Importance")
    ax.set_title(
        f"Tree-based Importance (Top {top_n}) - {len(result.tree_selected_features)} selected"
    )
    ax.legend()
    ax.invert_yaxis()

    # Linear importance - Top N (coloring based on linear_selected_features)
    ax = axes[0, 1]
    sorted_idx = np.argsort(result.mean_linear_importance)[::-1][:top_n]
    colors = [
        "green" if result.feature_names[i] in result.linear_selected_features else "red"
        for i in sorted_idx
    ]
    ax.barh(
        range(len(sorted_idx)), result.mean_linear_importance[sorted_idx], color=colors
    )
    ax.axvline(
        result.random_linear_importance,
        color="blue",
        linestyle="--",
        label="Random Feature",
    )
    ax.set_yticks(range(len(sorted_idx)))
    ax.set_yticklabels([result.feature_names[i][:30] for i in sorted_idx], fontsize=8)
    ax.set_xlabel("|Coefficient|")
    ax.set_title(
        f"Linear Importance (Top {top_n}) - {len(result.linear_selected_features)} selected"
    )
    ax.legend()
    ax.invert_yaxis()

    # Tree vs Linear scatter (coloring based on union selection)
    ax = axes[1, 0]
    colors = [
        "green" if f in result.selected_features else "red"
        for f in result.feature_names
    ]
    ax.scatter(
        result.mean_tree_importance, result.mean_linear_importance, c=colors, alpha=0.6
    )
    ax.axhline(result.random_linear_importance, color="blue", linestyle="--", alpha=0.5)
    ax.axvline(result.random_tree_importance, color="blue", linestyle="--", alpha=0.5)
    ax.set_xlabel("Tree Importance")
    ax.set_ylabel("Linear Importance")
    ax.set_title("Tree vs Linear Importance (Union)")

    # Summary pie with 3 categories
    ax = axes[1, 1]
    # Calculate: tree only, linear only, both, neither
    tree_set = set(result.tree_selected_features)
    linear_set = set(result.linear_selected_features)
    both = len(tree_set & linear_set)
    tree_only = len(tree_set - linear_set)
    linear_only = len(linear_set - tree_set)
    neither = len(result.dropped_features)

    sizes = [both, tree_only, linear_only, neither]
    labels = [
        f"Both ({both})",
        f"Tree only ({tree_only})",
        f"Linear only ({linear_only})",
        f"Dropped ({neither})",
    ]
    colors_pie = ["darkgreen", "lightgreen", "lightblue", "red"]
    ax.pie(sizes, labels=labels, colors=colors_pie, autopct="%1.1f%%", startangle=90)
    ax.set_title("Feature Selection Summary")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def select_features(X: pd.DataFrame, selected_features: List[str]) -> pd.DataFrame:
    """
    Sélectionne uniquement les features significatives.
    """
    return X[selected_features].copy()
