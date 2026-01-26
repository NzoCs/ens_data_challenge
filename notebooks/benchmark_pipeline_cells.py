# =============================================================================
# BENCHMARK SURVIVAL PIPELINE - Notebook Cells
# =============================================================================
# Copier chaque section (# %%) comme cellule dans le notebook
# =============================================================================


# %%
# =============================================================================
# CELL 1: CONFIGURATION & IMPORTS
# =============================================================================

import numpy as np
import pandas as pd
import optuna
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

# Configuration - À MODIFIER ICI
N_FOLDS = 5
N_TRIALS = 30
RANDOM_STATE = 42

# Modèles à utiliser
CLF_MODELS = ["XGB", "LGBM", "CatBoost", "RF", "LR"]
REG_MODELS = ["XGB", "LGBM", "CatBoost", "Ridge", "ElasticNet"]
IPCW_MODELS = ["XGB", "LGBM", "CatBoost"]

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Imports du module training
from ens_data_challenge.training import (
    transform_y,
    scale_01,
    FoldMetrics,
    CVResults,
    compute_ipcw_cindex,
    get_classifier_factory,
    get_regressor_factory,
    train_classifier_cv,
    train_regressor_cv,
    train_ensemble,
    predict_ensemble,
    get_ipcw_weights,
)
from ens_data_challenge.preprocess import Preprocessor

print("Configuration loaded!")
print(f"  N_FOLDS={N_FOLDS}, N_TRIALS={N_TRIALS}")


# %%
# =============================================================================
# CELL 2: DATA LOADING & PREPROCESSING
# =============================================================================

# Charger les données via globals
from ens_data_challenge.globals import (
    TRAIN_CLINICAL_DATA_PATH,
    TRAIN_MOLECULAR_DATA_PATH,
    TRAIN_TARGET_PATH,
    TEST_CLINICAL_DATA_PATH,
    TEST_MOLECULAR_DATA_PATH,
)
from ens_data_challenge.feature_engineering import FeatureEngineerHelper

clinical_train = pd.read_csv(TRAIN_CLINICAL_DATA_PATH)
clinical_test = pd.read_csv(TEST_CLINICAL_DATA_PATH)
molecular_train = pd.read_csv(TRAIN_MOLECULAR_DATA_PATH)
molecular_test = pd.read_csv(TEST_MOLECULAR_DATA_PATH)
targets_train = pd.read_csv(TRAIN_TARGET_PATH)

# Preprocessing
preprocessor = Preprocessor()

# Get cyto features
clinical_train, cyto_struct_train = preprocessor.get_cyto_features_and_df(
    clinical_train
)
clinical_test, cyto_struct_test = preprocessor.get_cyto_features_and_df(clinical_test)

# Fit transform
(
    clinical_preprocess_train,
    clinical_preprocess_test,
    molecular_preprocess_train,
    molecular_preprocess_test,
    cyto_struct_train,
    cyto_struct_test,
    targets_preprocess,
) = preprocessor.fit_transform(
    clinical_train,
    clinical_test,
    molecular_train,
    molecular_test,
    cyto_struct_train,
    cyto_struct_test,
    targets_train,
)

# =============================================================================
# FEATURE ENGINEERING avec FeatureEngineerHelper
# =============================================================================
feat_helper = FeatureEngineerHelper()

# 1. Ajouter Nmut (nombre de mutations par patient)
clinical_preprocess_train = feat_helper.Nmut(
    molecular_preprocess_train, clinical_preprocess_train
)
clinical_preprocess_test = feat_helper.Nmut(
    molecular_preprocess_test, clinical_preprocess_test
)

# 2. Ajouter ratios et interactions (WBC/ANC, PLT/HB, blast_cyto_complexity, tumor_burden_composite)
clinical_preprocess_train = feat_helper.ratios_and_interactions(
    clinical_preprocess_train
)
clinical_preprocess_test = feat_helper.ratios_and_interactions(clinical_preprocess_test)

# 3. Ajouter severity scores (cytopenias_count)
clinical_preprocess_train = feat_helper.severity_scores(clinical_preprocess_train)
clinical_preprocess_test = feat_helper.severity_scores(clinical_preprocess_test)

# 4. Ajouter encodage moléculaire par PATHWAY (confidence_weighted avec effect weighting)
clinical_preprocess_train = feat_helper.add_mol_encoding(
    clinical_data=clinical_preprocess_train,
    molecular_data=molecular_preprocess_train,
    col="PATHWAY",
    method="confidence_weighted",
    apply_effect_weighting=True,
)
clinical_preprocess_test = feat_helper.add_mol_encoding(
    clinical_data=clinical_preprocess_test,
    molecular_data=molecular_preprocess_test,
    col="PATHWAY",
    method="confidence_weighted",
    apply_effect_weighting=True,
)

# 5. Ajouter encodage moléculaire par GENE (constant pour baseline)
clinical_preprocess_train = feat_helper.add_mol_encoding(
    clinical_data=clinical_preprocess_train,
    molecular_data=molecular_preprocess_train,
    col="GENE",
    method="constant",
    apply_effect_weighting=False,
)
clinical_preprocess_test = feat_helper.add_mol_encoding(
    clinical_data=clinical_preprocess_test,
    molecular_data=molecular_preprocess_test,
    col="GENE",
    method="constant",
    apply_effect_weighting=False,
)

# Prepare data
drop_columns = ["ID", "OS_YEARS", "OS_STATUS", "EFS_YEARS", "EFS_STATUS"]
y_times = targets_preprocess["OS_YEARS"].values
events = targets_preprocess["OS_STATUS"].values

X_clinical = (
    clinical_preprocess_train.drop(columns=drop_columns, errors="ignore")
    .copy()
    .fillna(0)
    .replace(
        [np.inf, -np.inf], 0
    )  # Remplacer les inf créés par les divisions (ex: WBC/ANC quand ANC=0)
)

# Supprimer les colonnes catégorielles (garder uniquement les colonnes numériques)
categorical_cols = X_clinical.select_dtypes(
    include=["object", "category"]
).columns.tolist()
if categorical_cols:
    print(f"Dropping categorical columns: {categorical_cols}")
    X_clinical = X_clinical.drop(columns=categorical_cols)

X_cyto = cyto_struct_train.drop(columns=["ID"], errors="ignore").fillna("UNKNOWN")

print(f"Data loaded!")
print(f"  X_clinical: {X_clinical.shape}")
print(f"  X_cyto: {X_cyto.shape}")
print(f"  Event rate: {events.mean():.3f}")
print(f"\nFeature Engineering ajouté:")
print(f"  - Nmut (nombre de mutations)")
print(
    f"  - Ratios: wbc_anc_ratio, plt_hb_ratio, blast_cyto_complexity, tumor_burden_composite"
)
print(f"  - Severity: cytopenias_count")
print(
    f"  - Molecular encoding: PATHWAY (confidence_weighted + effect), GENE (constant)"
)


# %%
# =============================================================================
# CELL 3: FEATURE IMPORTANCE PRETRAINING
# =============================================================================
# Analyse feature importance avec random feature baseline
# Sélectionne uniquement les features significatives
# =============================================================================

from ens_data_challenge.training import (
    analyze_feature_importance,
    plot_feature_importance,
    select_features,
)

print("=" * 60)
print("FEATURE IMPORTANCE PRETRAINING")
print("=" * 60)

# Classification task - P(event=1)
print("\n--- Classification Feature Importance ---")
fi_clf = analyze_feature_importance(
    X_clinical,
    events,
    is_classifier=True,
    threshold_method="random",
    random_state=RANDOM_STATE,
    verbose=True,
)

# Regression task (using transformed y)
from ens_data_challenge.training import transform_y

y_transformed = transform_y(y_times)

print("\n--- Regression Feature Importance ---")
fi_reg = analyze_feature_importance(
    X_clinical,
    y_transformed,
    is_classifier=False,
    threshold_method="random",
    random_state=RANDOM_STATE,
    verbose=True,
)

# Keep features significant in BOTH classification AND regression
selected_clf = set(fi_clf.selected_features)
selected_reg = set(fi_reg.selected_features)
final_selected = list(
    selected_clf | selected_reg
)  # Union pour garder les features importants

# Log dropped features
dropped_features = [f for f in X_clinical.columns if f not in final_selected]
print(f"\n{'=' * 60}")
print(f"FINAL FEATURE SELECTION")
print(f"{'=' * 60}")
print(f"Original features: {len(X_clinical.columns)}")
print(f"Selected features: {len(final_selected)}")
print(f"Dropped features: {len(dropped_features)}")
print(
    f"\nDropped: {dropped_features[:20]}..."
    if len(dropped_features) > 20
    else f"\nDropped: {dropped_features}"
)

# Plot
fig_clf = plot_feature_importance(fi_clf, top_n=25)
fig_clf.suptitle("Classification Feature Importance", fontsize=14)
plt.show()

fig_reg = plot_feature_importance(fi_reg, top_n=25)
fig_reg.suptitle("Regression Feature Importance", fontsize=14)
plt.show()

# Apply selection
X_clinical_selected = select_features(X_clinical, final_selected)
print(f"\nX_clinical after selection: {X_clinical_selected.shape}")


# %%
# =============================================================================
# CELL 4: CLASSIFICATION - P(event=1)
# =============================================================================

print("=" * 60)
print("CLASSIFICATION: P(event=1)")
print("=" * 60)

# Use selected features
X_train = X_clinical_selected

# Train Ensemble
clf_results_dict = train_ensemble(
    X_train,
    y_times,
    events,
    CLF_MODELS,
    is_classifier=True,
    n_folds=N_FOLDS,
    n_trials=N_TRIALS,
    random_state=RANDOM_STATE,
)

clf_results = clf_results_dict["individual"]
oof_proba = clf_results_dict["ensemble_oof"]  # Optimized ensemble OOF

print(f"\n{'=' * 60}")
print(f"ENSEMBLE P(event=1)")
final_auc = compute_ipcw_cindex(y_times, events, y_times, events, oof_proba)
print(f"  Ensemble IPCW C-index: {final_auc:.4f}")
print(f"  Ensemble Weights: {clf_results_dict['ensemble_weights']}")


# %%
# =============================================================================
# CELL 4: REGRESSION - E[rank|event=1]
# =============================================================================

print("\n" + "=" * 60)
print("REGRESSION: E[rank|event=1] (trained on events only)")
print("=" * 60)

e1_mask = events == 1
e1_mask = events == 1
reg_e1_results_dict = train_ensemble(
    X_train,
    y_times,
    events,
    REG_MODELS,
    is_classifier=False,
    subset_mask=e1_mask,
    n_folds=N_FOLDS,
    n_trials=N_TRIALS,
    random_state=RANDOM_STATE,
)

reg_e1_results = reg_e1_results_dict["individual"]
oof_rank_e1 = reg_e1_results_dict["ensemble_oof"]
print(f"  Ensemble Weights: {reg_e1_results_dict['ensemble_weights']}")


# %%
# =============================================================================
# CELL 5: REGRESSION - E[rank|event=0]
# =============================================================================

print("\n" + "=" * 60)
print("REGRESSION: E[rank|event=0] (trained on censored only)")
print("=" * 60)

e0_mask = events == 0
e0_mask = events == 0
reg_e0_results_dict = train_ensemble(
    X_train,
    y_times,
    events,
    REG_MODELS,
    is_classifier=False,
    subset_mask=e0_mask,
    n_folds=N_FOLDS,
    n_trials=N_TRIALS,
    random_state=RANDOM_STATE,
)

reg_e0_results = reg_e0_results_dict["individual"]
oof_rank_e0 = reg_e0_results_dict["ensemble_oof"]
print(f"  Ensemble Weights: {reg_e0_results_dict['ensemble_weights']}")


# %%
# =============================================================================
# CELL 6: REGRESSION - E[rank|IPCW]
# =============================================================================

print("\n" + "=" * 60)
print("REGRESSION: E[rank|IPCW] (trained with KM weights)")
print("=" * 60)

reg_ipcw_results_dict = train_ensemble(
    X_train,
    y_times,
    events,
    IPCW_MODELS,
    is_classifier=False,
    use_ipcw=True,
    n_folds=N_FOLDS,
    n_trials=N_TRIALS,
    random_state=RANDOM_STATE,
)

reg_ipcw_results = reg_ipcw_results_dict["individual"]
oof_rank_ipcw = reg_ipcw_results_dict["ensemble_oof"]
print(f"  Ensemble Weights: {reg_ipcw_results_dict['ensemble_weights']}")


# %%
# =============================================================================
# CELL 7: CYTO STRUCT CLASSIFICATION (CatBoost avec cat_features)
# =============================================================================

from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold

print("\n" + "=" * 60)
print("CYTO STRUCT: CatBoost Classification")
print("=" * 60)

cat_features = list(X_cyto.columns)
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

oof_proba_cyto = np.zeros(len(events))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_cyto, events)):
    train_pool = Pool(
        X_cyto.iloc[train_idx], events[train_idx], cat_features=cat_features
    )
    val_pool = Pool(X_cyto.iloc[val_idx], events[val_idx], cat_features=cat_features)

    model = CatBoostClassifier(
        iterations=200, depth=6, learning_rate=0.1, random_state=RANDOM_STATE, verbose=0
    )
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    oof_proba_cyto[val_idx] = model.predict_proba(X_cyto.iloc[val_idx])[:, 1]

    print(
        f"  Fold {fold + 1}: AUC={compute_ipcw_cindex(y_times[train_idx], events[train_idx], y_times[val_idx], events[val_idx], oof_proba_cyto[val_idx]):.4f}"
    )


# %%
# =============================================================================
# CELL 8: META-FEATURES DATAFRAME
# =============================================================================

print("\n" + "=" * 60)
print("META-FEATURES")
print("=" * 60)

# Combine features
X_meta = X_train.copy()

# Predictions
X_meta["prob_event1"] = oof_proba
X_meta["prob_event0"] = 1 - oof_proba
X_meta["prob_cyto"] = oof_proba_cyto
X_meta["rank_e1"] = scale_01(oof_rank_e1)
X_meta["rank_e0"] = scale_01(oof_rank_e0)
X_meta["rank_ipcw"] = scale_01(oof_rank_ipcw)

# Polynomial features
pred_cols = ["prob_event1", "rank_e1", "rank_e0", "rank_ipcw"]
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_feats = poly.fit_transform(X_meta[pred_cols].values)
for i in range(poly_feats.shape[1]):
    X_meta[f"poly_{i}"] = poly_feats[:, i]

print(f"Meta-features: {X_meta.shape}")


# %%
# =============================================================================
# CELL 9: FORMULA OPTIMIZATION
# =============================================================================

from sklearn.model_selection import KFold

print("\n" + "=" * 60)
print("FORMULA OPTIMIZATION")
print("=" * 60)
print(
    "Rang = P(e=0)*w0*E[r|e=0] + P(e=1)*(w1_base + w1_rank*E[r|e=1]) + w_ipcw*E[IPCW]"
)

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

# Prepare scaled values
proba = oof_proba
rank_e0 = scale_01(oof_rank_e0)
rank_e1 = scale_01(oof_rank_e1)
rank_ipcw = scale_01(oof_rank_ipcw)


def formula_objective(trial):
    w0 = trial.suggest_float("w0", 0.0, 1.0)
    w1_base = trial.suggest_float("w1_base", 0.0, 1.0)
    w1_rank = trial.suggest_float("w1_rank", 0.0, 1.0)
    w_ipcw = trial.suggest_float("w_ipcw", 0.0, 1.0)

    c_indices = []
    for train_idx, val_idx in kf.split(proba):
        prob_e0 = 1 - proba[val_idx]
        prob_e1 = proba[val_idx]

        risk = (
            prob_e0 * w0 * rank_e0[val_idx]
            + prob_e1 * (w1_base + w1_rank * rank_e1[val_idx])
            + w_ipcw * rank_ipcw[val_idx]
        )
        risk_scores = pd.Series(risk).rank().values / len(risk)

        c_idx = compute_ipcw_cindex(
            y_times[train_idx],
            events[train_idx],
            y_times[val_idx],
            events[val_idx],
            risk_scores,
        )
        c_indices.append(c_idx)

    return -np.mean(c_indices)


study = optuna.create_study(direction="minimize")
study.optimize(formula_objective, n_trials=100, show_progress_bar=True)

formula_params = study.best_params
formula_score = -study.best_value

print(f"\nBest params: {formula_params}")
print(f"Formula IPCW C-index: {formula_score:.4f}")

# Apply formula
prob_e0 = 1 - proba
prob_e1 = proba
formula_risk = (
    prob_e0 * formula_params["w0"] * rank_e0
    + prob_e1 * (formula_params["w1_base"] + formula_params["w1_rank"] * rank_e1)
    + formula_params["w_ipcw"] * rank_ipcw
)
formula_risk = pd.Series(formula_risk).rank().values / len(formula_risk)


# %%
# =============================================================================
# CELL 10: META-MODELS
# =============================================================================

print("\n" + "=" * 60)
print("META-MODELS")
print("=" * 60)

meta_model_names = ["XGB", "LGBM", "Ridge"]

meta_results_dict = train_ensemble(
    X_meta,
    y_times,
    events,
    meta_model_names,
    is_classifier=False,
    use_ipcw=True,
    n_folds=N_FOLDS,
    n_trials=N_TRIALS,
    random_state=RANDOM_STATE,
)

meta_results = meta_results_dict["individual"]
oof_meta = meta_results_dict["ensemble_oof"]
print(f"  Meta Ensemble Weights: {meta_results_dict['ensemble_weights']}")

meta_risk = inverse_transform_pred(oof_meta)


# %%
# =============================================================================
# CELL 11: FINAL ENSEMBLE
# =============================================================================

print("\n" + "=" * 60)
print("FINAL ENSEMBLE: Formula + Meta-Models")
print("=" * 60)


def ensemble_objective(trial):
    w_formula = trial.suggest_float("w_formula", 0.0, 1.0)
    w_meta = trial.suggest_float("w_meta", 0.0, 1.0)

    w_sum = w_formula + w_meta + 1e-10
    combined = (w_formula / w_sum) * formula_risk + (w_meta / w_sum) * meta_risk
    combined_ranked = pd.Series(combined).rank().values / len(combined)

    c_indices = []
    for train_idx, val_idx in kf.split(combined):
        c_idx = compute_ipcw_cindex(
            y_times[train_idx],
            events[train_idx],
            y_times[val_idx],
            events[val_idx],
            combined_ranked[val_idx],
        )
        c_indices.append(c_idx)

    return -np.mean(c_indices)


study = optuna.create_study(direction="minimize")
study.optimize(ensemble_objective, n_trials=50, show_progress_bar=True)

ensemble_params = study.best_params
ensemble_score = -study.best_value

print(f"\nEnsemble params: {ensemble_params}")
print(f"Final IPCW C-index: {ensemble_score:.4f}")

# Final predictions
w_sum = ensemble_params["w_formula"] + ensemble_params["w_meta"]
final_risk = (
    ensemble_params["w_formula"] / w_sum * formula_risk
    + ensemble_params["w_meta"] / w_sum * meta_risk
)
final_risk = pd.Series(final_risk).rank().values / len(final_risk)

print(f"\n" + "=" * 60)
print("PIPELINE COMPLETE!")
print(f"Final risk shape: {final_risk.shape}")
print("=" * 60)


# %%
# =============================================================================
# CELL 12: RESULTS SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)

# Create summary DataFrame
summary_data = []

# Classification
for name, r in clf_results.items():
    summary_data.append(
        {
            "Type": "Classification",
            "Model": name,
            "IPCW_C": r["cv_results"].mean_ipcw_c_index,
            "Std": r["cv_results"].std_ipcw_c_index,
        }
    )

# Regression E1
for name, r in reg_e1_results.items():
    summary_data.append(
        {
            "Type": "Reg E[r|e=1]",
            "Model": name,
            "IPCW_C": r["cv_results"].mean_ipcw_c_index,
            "Std": r["cv_results"].std_ipcw_c_index,
        }
    )

# Regression E0
for name, r in reg_e0_results.items():
    summary_data.append(
        {
            "Type": "Reg E[r|e=0]",
            "Model": name,
            "IPCW_C": r["cv_results"].mean_ipcw_c_index,
            "Std": r["cv_results"].std_ipcw_c_index,
        }
    )

# IPCW
for name, r in reg_ipcw_results.items():
    summary_data.append(
        {
            "Type": "Reg IPCW",
            "Model": name,
            "IPCW_C": r["cv_results"].mean_ipcw_c_index,
            "Std": r["cv_results"].std_ipcw_c_index,
        }
    )

# Meta-models
for name, r in meta_results.items():
    summary_data.append(
        {
            "Type": "Meta-Model",
            "Model": name,
            "IPCW_C": r["cv_results"].mean_ipcw_c_index,
            "Std": r["cv_results"].std_ipcw_c_index,
        }
    )

# Final
summary_data.append(
    {"Type": "Formula", "Model": "Optimized", "IPCW_C": formula_score, "Std": 0.0}
)
summary_data.append(
    {"Type": "FINAL", "Model": "Ensemble", "IPCW_C": ensemble_score, "Std": 0.0}
)

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))
