# =============================================================================
# TRAINERS - Training loops with proper metrics reporting
# =============================================================================

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn.model_selection import KFold, StratifiedKFold
import optuna

from .transforms import transform_y
from .metrics import (
    FoldMetrics,
    CVResults,
    compute_ipcw_cindex,
    compute_mse,
    compute_auc,
)
from .model_factories import get_classifier_factory, get_regressor_factory
from .utils.kaplan_meier_weights import get_ipcw_weights

# =============================================================================
# REGULARIZED OBJECTIVES HELPERS
# =============================================================================

# Modèles linéaires vs tree-based
LINEAR_MODELS = {
    "LR",
    "Ridge",
    "ElasticNet",
    "LDA",
    "QDA",
    "LinearSVC",
    "ElasticNetLR",
    "PLS",
    "KernelRidge",
}
TREE_MODELS = {"XGB", "LGBM", "CatBoost", "RF"}

# Coefficient de régularisation pour la complexité
ALPHA_TREE_COMPLEXITY = 0.001  # Pénalité pour nb feuilles
ALPHA_OVERFITTING_PENALTY = 0.1  # Pénalité pour l'écart train/val
ALPHA_LINEAR_COMPLEXITY = (
    0.01  # Pénalité pour la complexité linéaire (fonction de C, alpha)
)


def get_tree_complexity(model, model_name: str) -> float:
    """
    Estime la complexité d'un modèle arbre (approximation du nombre de feuilles).
    """
    try:
        if model_name == "XGB":
            # XGBoost: n_estimators * 2^max_depth (approximation)
            booster = model.get_booster()
            trees_df = booster.trees_to_dataframe()
            n_leaves = (trees_df["Feature"] == "Leaf").sum()
            return n_leaves
        elif model_name == "LGBM":
            # LightGBM: num_leaves * n_estimators
            return model.n_estimators_ * model.get_params().get("num_leaves", 31)
        elif model_name == "CatBoost":
            # CatBoost: approximation via tree_count * 2^depth
            return model.tree_count_ * (2 ** model.get_param("depth"))
        elif model_name == "RF":
            # Random Forest: sum of n_leaves across all trees
            total_leaves = sum(tree.get_n_leaves() for tree in model.estimators_)
            return total_leaves
    except Exception:
        pass
    # Fallback: use n_estimators or iterations as proxy
    if hasattr(model, "n_estimators_"):
        return model.n_estimators_ * 10
    if hasattr(model, "tree_count_"):
        return model.tree_count_ * 10
    return 100  # Default


def get_linear_complexity(model, model_name: str) -> float:
    """
    Retourne une mesure de complexité pour un modèle linéaire basée sur sa régularisation.
    Plus la régularisation est faible (grand C, petit alpha), plus la complexité est grande.
    """
    try:
        # Check if wrapped in Pipeline
        if hasattr(model, "steps"):
            # Assume last step is the estimator
            model = model.steps[-1][1]

        if model_name in ["LR", "ElasticNetLR"]:
            # LogisticRegression: C = 1/lambda
            c_val = model.get_params().get("C", 1.0)
            return np.log1p(c_val)
        elif model_name == "LinearSVC":
            # LinearSVC wrapped in CalibratedClassifierCV
            # CalibratedClassifierCV.base_estimator is the LinearSVC (if not fitting)
            # But after fitting, it has calibrated_classifiers_
            # We check if we can access the base estimator's C
            if hasattr(model, "base_estimator"):
                c_val = model.base_estimator.get_params().get("C", 1.0)
                return np.log1p(c_val)
            return 1.0
        elif model_name in ["Ridge", "ElasticNet"]:
            alpha_val = model.alpha
            return np.log1p(1.0 / (alpha_val + 1e-10))
        elif model_name == "KernelRidge":
            alpha_val = model.alpha
            return np.log1p(1.0 / (alpha_val + 1e-10))
        elif model_name == "PLS":
            # More components = more complex
            return float(model.n_components)
        elif model_name in ["LDA", "QDA"]:
            # LDA/QDA: complexity is constantish or depends on shrinkage/reg_param
            # Let's simplify and return 1.0 or small penalty
            return 1.0
    except Exception:
        pass
    return 1.0


def train_classifier_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    y_times: np.ndarray,
    events: np.ndarray,
    model_name: str,
    n_folds: int = 5,
    n_trials: int = 30,
    random_state: int = 42,
    verbose: bool = True,
) -> Tuple[np.ndarray, List, CVResults, Dict[str, Any]]:
    """
    Entraîne un classifieur avec Optuna et cross-validation.

    Returns:
        oof_preds: Out-of-fold predictions
        models: List of trained models
        cv_results: CVResults with detailed metrics
        best_params: Best Optuna parameters
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    factory = get_classifier_factory(model_name, random_state)

    def objective(trial):
        fold_scores = []
        fold_train_scores = []
        fold_complexities = []

        for train_idx, val_idx in skf.split(X, y):
            model = factory(trial)
            model.fit(X.iloc[train_idx], y[train_idx])

            # Train evaluation
            train_pred = model.predict_proba(X.iloc[train_idx])[:, 1]
            train_auc = compute_auc(y[train_idx], train_pred)
            fold_train_scores.append(train_auc)

            # Val evaluation
            pred = model.predict_proba(X.iloc[val_idx])[:, 1]
            auc = compute_auc(y[val_idx], pred)
            fold_scores.append(auc)

            # Calcul de la complexité selon le type de modèle
            if model_name in TREE_MODELS:
                complexity = get_tree_complexity(model, model_name)
                fold_complexities.append(complexity)
            elif model_name in LINEAR_MODELS:
                # Pour classification linéaire: complexité basée sur la régularisation
                complexity = get_linear_complexity(model, model_name)
                fold_complexities.append(complexity)

        mean_auc = np.mean(fold_scores)
        mean_train_auc = np.mean(fold_train_scores)

        # Penalize overfitting: maximize (Val - ALPHA * |Val - Train|)
        gap = abs(mean_auc - mean_train_auc)
        score = mean_auc - ALPHA_OVERFITTING_PENALTY * gap

        # Objectif régularisé
        if model_name in TREE_MODELS:
            # loss + alpha * nb_feuilles
            mean_complexity = np.mean(fold_complexities)
            regularized_loss = -score + ALPHA_TREE_COMPLEXITY * mean_complexity
            return regularized_loss

        elif model_name in LINEAR_MODELS:
            # loss + alpha * complexité_linéaire
            mean_complexity = np.mean(fold_complexities)
            regularized_loss = -score + ALPHA_LINEAR_COMPLEXITY * mean_complexity
            return regularized_loss
        else:
            return -score

    # Optimize
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=verbose)

    # Train final avec best params
    oof = np.zeros(len(y))
    models = []
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        trial = optuna.trial.FixedTrial(study.best_params)
        model = factory(trial)
        model.fit(X.iloc[train_idx], y[train_idx])

        # Predictions
        train_pred = model.predict_proba(X.iloc[train_idx])[:, 1]
        val_pred = model.predict_proba(X.iloc[val_idx])[:, 1]
        oof[val_idx] = val_pred

        # Metrics
        train_loss = -compute_auc(y[train_idx], train_pred)
        val_loss = -compute_auc(y[val_idx], val_pred)
        auc = compute_auc(y[val_idx], val_pred)

        # IPCW C-index
        risk_scores = val_pred  # Higher prob = higher risk
        ipcw_c = compute_ipcw_cindex(
            y_times[train_idx],
            events[train_idx],
            y_times[val_idx],
            events[val_idx],
            risk_scores,
        )

        fold_metrics.append(
            FoldMetrics(
                fold=fold,
                train_loss=train_loss,
                val_loss=val_loss,
                c_index=auc,
                ipcw_c_index=ipcw_c,
                auc=auc,
            )
        )
        models.append(model)

        if verbose:
            print(f"  Fold {fold + 1}: AUC={auc:.4f}, IPCW_C={ipcw_c:.4f}")

    cv_results = CVResults.from_folds(fold_metrics)
    if verbose:
        print(cv_results.summary())

    return oof, models, cv_results, study.best_params


def train_regressor_cv(
    X: pd.DataFrame,
    y_times: np.ndarray,
    events: np.ndarray,
    model_name: str,
    subset_mask: Optional[np.ndarray] = None,
    use_ipcw: bool = False,
    n_folds: int = 5,
    n_trials: int = 30,
    random_state: int = 42,
    verbose: bool = True,
) -> Tuple[np.ndarray, List, CVResults, Dict[str, Any]]:
    """
    Entraîne un régresseur avec Optuna et cross-validation.

    Args:
        subset_mask: Boolean mask to filter training data (e.g., events==1)
        use_ipcw: Use Kaplan-Meier IPCW weights for training

    Returns:
        oof_preds, models, cv_results, best_params
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    factory = get_regressor_factory(model_name, random_state)

    def objective(trial):
        fold_mses = []
        fold_train_mses = []
        fold_complexities = []
        fold_n_samples = []

        for train_idx, val_idx in kf.split(X):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            times_tr = y_times[train_idx]
            y_tr = transform_y(times_tr)
            y_val = transform_y(y_times[val_idx])

            # Apply subset mask
            if subset_mask is not None:
                mask = subset_mask[train_idx]
                if mask.sum() < 10:
                    continue
                X_tr = X_tr.iloc[mask.nonzero()[0]]
                y_tr = y_tr[mask]

            model = factory(trial)

            # Apply IPCW weights
            if use_ipcw and subset_mask is None:
                weights = get_ipcw_weights(times_tr, events[train_idx]).values
                try:
                    model.fit(X_tr, y_tr, sample_weight=weights)
                except TypeError:
                    model.fit(X_tr, y_tr)
            else:
                model.fit(X_tr, y_tr)

            # Train MSE
            train_pred = model.predict(X_tr)
            train_mse = compute_mse(y_tr, train_pred)
            fold_train_mses.append(train_mse)

            # Val MSE
            pred = model.predict(X_val)
            mse = compute_mse(y_val, pred)
            fold_mses.append(mse)
            fold_n_samples.append(len(y_val))

            # Calcul de la complexité selon le type de modèle
            if model_name in TREE_MODELS:
                complexity = get_tree_complexity(model, model_name)
                fold_complexities.append(complexity)
            elif model_name in LINEAR_MODELS:
                complexity = get_linear_complexity(model, model_name)
                fold_complexities.append(complexity)

        if not fold_mses:
            return 1.0

        mean_mse = np.mean(fold_mses)
        mean_train_mse = np.mean(fold_train_mses)

        # Penalize overfitting: minimize (Val + ALPHA * |Val - Train|)
        gap = abs(mean_mse - mean_train_mse)
        score = mean_mse + ALPHA_OVERFITTING_PENALTY * gap

        # Objectif régularisé
        if model_name in TREE_MODELS:
            # MSE + alpha * nb_feuilles
            mean_complexity = np.mean(fold_complexities)
            regularized_loss = score + ALPHA_TREE_COMPLEXITY * mean_complexity
            return regularized_loss
        elif model_name in LINEAR_MODELS:
            # MSE + alpha * complexité_linéaire
            mean_complexity = np.mean(fold_complexities)
            regularized_loss = score + ALPHA_LINEAR_COMPLEXITY * mean_complexity
            return regularized_loss
        else:
            return score

    # Optimize
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=verbose)

    # Train final
    oof = np.zeros(len(y_times))
    models = []
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        times_tr, times_val = y_times[train_idx], y_times[val_idx]
        events_tr, events_val = events[train_idx], events[val_idx]
        y_tr = transform_y(times_tr)
        y_val = transform_y(times_val)

        # Apply subset
        if subset_mask is not None:
            mask = subset_mask[train_idx]
            if mask.sum() < 10:
                continue
            X_tr_subset = X_tr.iloc[mask.nonzero()[0]]
            y_tr_subset = y_tr[mask]
        else:
            X_tr_subset = X_tr
            y_tr_subset = y_tr

        trial = optuna.trial.FixedTrial(study.best_params)
        model = factory(trial)

        if use_ipcw and subset_mask is None:
            weights = get_ipcw_weights(times_tr, events_tr).values
            try:
                model.fit(X_tr_subset, y_tr_subset, sample_weight=weights)
            except TypeError:
                model.fit(X_tr_subset, y_tr_subset)
        else:
            model.fit(X_tr_subset, y_tr_subset)

        # Predictions
        train_pred = model.predict(X_tr)
        val_pred = model.predict(X_val)
        oof[val_idx] = val_pred

        # Metrics
        train_loss = compute_mse(y_tr, train_pred)
        val_loss = compute_mse(y_val, val_pred)

        # IPCW C-index
        risk_scores = val_pred
        ipcw_c = compute_ipcw_cindex(
            times_tr, events_tr, times_val, events_val, risk_scores
        )

        fold_metrics.append(
            FoldMetrics(
                fold=fold, train_loss=train_loss, val_loss=val_loss, ipcw_c_index=ipcw_c
            )
        )
        models.append(model)

        if verbose:
            print(f"  Fold {fold + 1}: MSE={val_loss:.4f}, IPCW_C={ipcw_c:.4f}")

    cv_results = CVResults.from_folds(fold_metrics)
    if verbose:
        print(cv_results.summary())

    return oof, models, cv_results, study.best_params


def train_ensemble(
    X: pd.DataFrame,
    y_times: np.ndarray,
    events: np.ndarray,
    model_names: List[str],
    is_classifier: bool = True,
    subset_mask: Optional[np.ndarray] = None,
    use_ipcw: bool = False,
    n_folds: int = 5,
    n_trials: int = 30,
    random_state: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Optimise les poids de l'ensemble pour maximiser IPCW C-index.
    """

    def optimize_valid_ensemble_weights(
        predictions: Dict[str, np.ndarray],
        y_times: np.ndarray,
        events: np.ndarray,
        n_trials: int = 50,
    ) -> Dict[str, float]:
        model_keys = list(predictions.keys())

        def objective(trial):
            weights = []
            for k in model_keys:
                w = trial.suggest_float(f"w_{k}", 0.0, 1.0)
                weights.append(w)

            # Normalize
            total = sum(weights)
            if total == 0:
                return 0.0
            weights = [w / total for w in weights]

            # Weighted average
            ensemble_pred = np.zeros_like(predictions[model_keys[0]])
            for k, w in zip(model_keys, weights):
                ensemble_pred += w * predictions[k]

            # Score
            try:
                score = compute_ipcw_cindex(
                    y_times, events, y_times, events, ensemble_pred
                )
            except Exception:
                score = 0.5

            return score  # Maximize

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        # Best weights
        best_w = []
        for k in model_keys:
            best_w.append(study.best_params.get(f"w_{k}", 0.0))
        total = sum(best_w)
        if total == 0:
            return {k: 1.0 / len(model_keys) for k in model_keys}

        return {k: w / total for k in zip(model_keys, best_w)}

    results = {}

    for model_name in model_names:
        if verbose:
            print(f"\n{'=' * 40}")
            print(f"Training {model_name}...")
            print(f"{'=' * 40}")

        if is_classifier:
            oof, models, cv_results, params = train_classifier_cv(
                X,
                events,
                y_times,
                events,
                model_name,
                n_folds=n_folds,
                n_trials=n_trials,
                random_state=random_state,
                verbose=verbose,
            )
        else:
            oof, models, cv_results, params = train_regressor_cv(
                X,
                y_times,
                events,
                model_name,
                subset_mask=subset_mask,
                use_ipcw=use_ipcw,
                n_folds=n_folds,
                n_trials=n_trials,
                random_state=random_state,
                verbose=verbose,
            )

        results[model_name] = {
            "oof": oof,
            "models": models,
            "cv_results": cv_results,
            "params": params,
        }

    # Optimize ensemble weights
    if verbose:
        print("\nOptimizing ensemble weights...")

    oof_preds = {k: v["oof"] for k, v in results.items()}

    # Simple mean
    ensemble_oof_mean = np.mean([p for p in oof_preds.values()], axis=0)

    # Weighted optimization
    def objective_weights(trial):
        weights = {}
        total_weight = 0

        ensemble_pred = np.zeros(len(y_times))

        for name in oof_preds.keys():
            w = trial.suggest_float(f"w_{name}", 0.0, 1.0)
            weights[name] = w
            total_weight += w
            ensemble_pred += w * oof_preds[name]

        if total_weight == 0:
            return 0.0

        ensemble_pred /= total_weight

        score = compute_ipcw_cindex(y_times, events, y_times, events, ensemble_pred)
        return score

    study_w = optuna.create_study(direction="maximize")
    study_w.optimize(objective_weights, n_trials=50, show_progress_bar=False)

    best_weights = study_w.best_params
    total_w = sum(best_weights.values())

    ensemble_oof_optimized = np.zeros(len(y_times))
    if total_w > 0:
        for name, w in best_weights.items():
            norm_w = w / total_w
            # Remove w_ prefix if present (it is present from suggest_float)
            real_name = name.replace("w_", "")
            if real_name in oof_preds:
                ensemble_oof_optimized += norm_w * oof_preds[real_name]
    else:
        ensemble_oof_optimized = ensemble_oof_mean

    return {
        "individual": results,
        "ensemble_oof": ensemble_oof_optimized,
        "ensemble_oof_mean": ensemble_oof_mean,
        "ensemble_weights": best_weights,
    }


def predict_ensemble(
    X: pd.DataFrame,
    ensemble_results: Dict[str, Any],
    is_classifier: bool = True,
) -> np.ndarray:
    """
    Fait des prédictions avec l'ensemble entraîné (moyenne pondérée des folds et des modèles).
    """
    weights = ensemble_results.get("ensemble_weights", {})
    individual_results = ensemble_results.get("individual", {})

    # Re-calculate total weight sum for normalization
    total_weight = 0.0
    model_weight_map = {}

    for model_name in individual_results.keys():
        w_key = f"w_{model_name}"
        if w_key in weights:
            w = weights[w_key]
        elif model_name in weights:
            w = weights[model_name]
        else:
            w = 0.0
        model_weight_map[model_name] = w
        total_weight += w

    if total_weight == 0:
        # Fallback to mean
        total_weight = len(individual_results)
        for k in model_weight_map:
            model_weight_map[k] = 1.0

    # Prediction loop
    final_pred = np.zeros(len(X))

    for model_name, info in individual_results.items():
        w = model_weight_map[model_name] / total_weight
        if w <= 1e-6:
            continue

        models = info["models"]
        # Average prediction across folds
        model_pred = np.zeros(len(X))
        for model in models:
            if is_classifier:
                model_pred += model.predict_proba(X)[:, 1]
            else:
                model_pred += model.predict(X)
        model_pred /= len(models)

        final_pred += w * model_pred

    return final_pred
