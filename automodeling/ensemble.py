from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor

from automodeling.base import AutoModelBase

import logging

logging.getLogger("optuna").setLevel(logging.WARNING)


class AutoRandomForestRegressor(AutoModelBase, BaseEstimator, RegressorMixin):
    def __init__(
            self,
            n_estimators=100,
            criterion='squared_error',
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.0,
            max_features=1.0,
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            bootstrap=True,
            oob_score=False,
            n_jobs=-1,
            random_state=42,
            ccp_alpha=0.0,
            max_samples=None,
            auto_scoring=None,
            auto_direction='minimize', 
            auto_timeout=60, 
            auto_n_trials=None, 
            auto_verbose=False,
            auto_use_scaler=False
        ):
        super().__init__(
            auto_scoring,
            auto_direction,
            auto_timeout,
            auto_n_trials,
            auto_verbose,
            auto_use_scaler
        )
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.ccp_alpha = ccp_alpha
        self.max_samples = max_samples
    
    def _get_model_params(self):
        return {
            'n_estimators': self.n_estimators,
            'criterion': self.criterion,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'min_weight_fraction_leaf': self.min_weight_fraction_leaf,
            'max_features': self.max_features,
            'max_leaf_nodes': self.max_leaf_nodes,
            'min_impurity_decrease': self.min_impurity_decrease,
            'bootstrap': self.bootstrap,
            'oob_score': self.oob_score,
            'n_jobs': self.n_jobs,
            'random_state': self.random_state,
            'ccp_alpha': self.ccp_alpha,
            'max_samples': self.max_samples
        }
    
    def _build_model(self, params):
        return RandomForestRegressor(**params)

    def _get_search_space(self, trial):
        bootstrap = trial.suggest_categorical('bootstrap', [True, False])

        if bootstrap:
            oob_score = trial.suggest_categorical('oob_score', [False, True])
        else:
            oob_score = False

        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'criterion': trial.suggest_categorical(
                'criterion', ['squared_error', 'absolute_error', 'friedman_mse', 'poisson']
            ),
            'max_depth': trial.suggest_int('max_depth', 3, 30),
            'min_samples_split': trial.suggest_float('min_samples_split', 0.001, 0.3),
            'min_samples_leaf': trial.suggest_float('min_samples_leaf', 0.001, 0.1),
            'min_weight_fraction_leaf': trial.suggest_float('min_weight_fraction_leaf', 0.0, 0.1),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None, 1.0, 0.3, 0.7]),
            'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 10, 100) if trial.suggest_categorical('use_max_leaf_nodes', [True, False]) else None,
            'min_impurity_decrease': trial.suggest_float('min_impurity_decrease', 0.0, 0.01),
            'bootstrap': bootstrap,
            'oob_score': oob_score,
            'n_jobs': -1,
            'random_state': trial.suggest_categorical('random_state', [42]),
            'ccp_alpha': trial.suggest_float('ccp_alpha', 0.0, 0.01),
            'max_samples': trial.suggest_float('max_samples', 0.3, 1.0) if bootstrap else None,
        }
        
        return params