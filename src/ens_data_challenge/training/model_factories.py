# =============================================================================
# MODEL FACTORIES - Optuna model factories for classifiers and regressors
# =============================================================================
import optuna
from typing import Callable


def get_classifier_factory(model_name: str, random_state: int = 42) -> Callable:
    """
    Retourne une factory pour créer un classifieur avec params Optuna.

    Models: XGB, LGBM, CatBoost, RF, LR
    """
    from xgboost import XGBClassifier
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.discriminant_analysis import (
        LinearDiscriminantAnalysis,
        QuadraticDiscriminantAnalysis,
    )
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    def factory(trial: optuna.Trial):
        if model_name == "XGB":
            return XGBClassifier(
                n_estimators=trial.suggest_int("n_estimators", 1, 50),
                max_depth=trial.suggest_int("max_depth", 1, 10),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                subsample=trial.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
                reg_lambda=trial.suggest_float("reg_lambda", 1e-4, 10, log=True),
                reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 10, log=True),
                random_state=random_state,
                verbosity=0,
                n_jobs=-1,
            )
        elif model_name == "LGBM":
            return LGBMClassifier(
                n_estimators=trial.suggest_int("n_estimators", 1, 50),
                max_depth=trial.suggest_int("max_depth", 1, 10),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                num_leaves=trial.suggest_int("num_leaves", 20, 150),
                reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 10, log=True),
                reg_lambda=trial.suggest_float("reg_lambda", 1e-4, 10, log=True),
                random_state=random_state,
                verbose=-1,
                n_jobs=-1,
            )
        elif model_name == "CatBoost":
            return CatBoostClassifier(
                iterations=trial.suggest_int("iterations", 1, 50),
                depth=trial.suggest_int("depth", 1, 10),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1e-4, 10, log=True),
                model_size_reg=trial.suggest_float(
                    "model_size_reg", 1e-4, 10, log=True
                ),
                random_state=random_state,
                verbose=0,
            )
        elif model_name == "RF":
            return RandomForestClassifier(
                n_estimators=trial.suggest_int("n_estimators", 100, 300),
                max_depth=trial.suggest_int("max_depth", 1, 10),
                min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
                random_state=random_state,
                n_jobs=-1,
            )
        elif model_name == "LR":
            return make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=trial.suggest_float("C", 1e-4, 100, log=True),
                    max_iter=1000,
                    random_state=random_state,
                ),
            )
        elif model_name == "LDA":
            solver = trial.suggest_categorical("solver", ["svd", "lsqr", "eigen"])
            if solver == "svd":
                shrinkage = None
            else:
                shrinkage = trial.suggest_float("shrinkage", 0.0, 1.0)
            return make_pipeline(
                StandardScaler(),
                LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage),
            )
        elif model_name == "QDA":
            reg_param = trial.suggest_float("reg_param", 0.0, 1.0)
            return make_pipeline(
                StandardScaler(), QuadraticDiscriminantAnalysis(reg_param=reg_param)
            )
        elif model_name == "LinearSVC":
            # LinearSVC doesn't support predict_proba, so we wrap it
            return make_pipeline(
                StandardScaler(),
                CalibratedClassifierCV(
                    LinearSVC(
                        C=trial.suggest_float("C", 1e-4, 100, log=True),
                        dual=False,
                        random_state=random_state,
                    )
                ),
            )
        elif model_name == "ElasticNetLR":
            return make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    penalty="elasticnet",
                    solver="saga",
                    l1_ratio=trial.suggest_float("l1_ratio", 0.1, 0.9),
                    C=trial.suggest_float("C", 1e-4, 100, log=True),
                    max_iter=1000,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            )
        else:
            raise ValueError(f"Unknown classifier: {model_name}")

    return factory


def get_regressor_factory(model_name: str, random_state: int = 42) -> Callable:
    """
    Retourne une factory pour créer un régresseur avec params Optuna.

    Models: XGB, LGBM, CatBoost, RF, Ridge, ElasticNet
    """
    from xgboost import XGBRegressor
    from catboost import CatBoostRegressor
    from lightgbm import LGBMRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge, ElasticNet
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    def factory(trial: optuna.Trial):
        if model_name == "XGB":
            return XGBRegressor(
                n_estimators=trial.suggest_int("n_estimators", 1, 50),
                max_depth=trial.suggest_int("max_depth", 1, 10),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                subsample=trial.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
                reg_lambda=trial.suggest_float("reg_lambda", 1e-4, 10, log=True),
                reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 10, log=True),
                random_state=random_state,
                verbosity=0,
                n_jobs=-1,
            )
        elif model_name == "LGBM":
            return LGBMRegressor(
                n_estimators=trial.suggest_int("n_estimators", 1, 50),
                max_depth=trial.suggest_int("max_depth", 1, 12),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                num_leaves=trial.suggest_int("num_leaves", 20, 150),
                random_state=random_state,
                reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 10, log=True),
                reg_lambda=trial.suggest_float("reg_lambda", 1e-4, 10, log=True),
                verbose=-1,
                n_jobs=-1,
            )
        elif model_name == "CatBoost":
            return CatBoostRegressor(
                iterations=trial.suggest_int("iterations", 1, 50),
                depth=trial.suggest_int("depth", 1, 10),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                random_state=random_state,
                l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1e-4, 10, log=True),
                model_size_reg=trial.suggest_float(
                    "model_size_reg", 1e-4, 10, log=True
                ),
                verbose=0,
            )
        elif model_name == "RF":
            return RandomForestRegressor(
                n_estimators=trial.suggest_int("n_estimators", 100, 300),
                max_depth=trial.suggest_int("max_depth", 1, 10),
                min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
                random_state=random_state,
                n_jobs=-1,
            )
        elif model_name == "Ridge":
            return make_pipeline(
                StandardScaler(),
                Ridge(alpha=trial.suggest_float("alpha", 1e-4, 100, log=True)),
            )
        elif model_name == "ElasticNet":
            return make_pipeline(
                StandardScaler(),
                ElasticNet(
                    alpha=trial.suggest_float("alpha", 1e-4, 10, log=True),
                    l1_ratio=trial.suggest_float("l1_ratio", 0.1, 0.9),
                    max_iter=5000,
                ),
            )
        elif model_name == "PLS":
            return make_pipeline(
                StandardScaler(),
                PLSRegression(n_components=trial.suggest_int("n_components", 1, 10)),
            )
        elif model_name == "KernelRidge":
            kernel = trial.suggest_categorical(
                "kernel", ["linear", "poly", "rbf", "sigmoid"]
            )
            alpha = trial.suggest_float("alpha", 1e-4, 10, log=True)
            gamma = trial.suggest_float("gamma", 1e-4, 10, log=True)
            degree = trial.suggest_int("degree", 2, 5)

            return make_pipeline(
                StandardScaler(),
                KernelRidge(alpha=alpha, kernel=kernel, gamma=gamma, degree=degree),
            )
        else:
            raise ValueError(f"Unknown regressor: {model_name}")

    return factory
