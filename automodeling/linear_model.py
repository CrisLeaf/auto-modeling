from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, RegressorMixin

from automodeling.base import AutoModelBase

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor

import logging
import optuna

logging.getLogger("optuna").setLevel(logging.WARNING)


class AutoLinearRegression(AutoModelBase, BaseEstimator, RegressorMixin):
    def __init__(
            self,
            fit_intercept=True,
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
        self.fit_intercept = fit_intercept
    
    def _get_model_params(self):
        return {
            'fit_intercept': self.fit_intercept,
        }
    
    def _build_model(self, params):
        return LinearRegression(**params)
    
    def _get_search_space(self, trial):
        params = {
            'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
        }
        
        return params


class AutoRidge(AutoModelBase, BaseEstimator, RegressorMixin):
    def __init__(
            self, 
            alpha=1.0,
            fit_intercept=True,
            copy_X=True,
            max_iter=None,
            tol=1e-4,
            solver='auto',
            positive=False,
            random_state=None,
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
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X
        self.max_iter = max_iter
        self.tol = tol
        self.solver = solver
        self.positive = positive
        self.random_state = random_state
    
    def _get_model_params(self):
        return {
            'alpha': self.alpha,
            'fit_intercept': self.fit_intercept,
            'copy_X': self.copy_X,
            'max_iter': self.max_iter,
            'tol': self.tol,
            'solver': self.solver,
            'positive': self.positive,
            'random_state': self.random_state,
        }
    
    def _build_model(self, params):
        return Ridge(**params)
    
    def _get_search_space(self, trial):
        solver = trial.suggest_categorical('solver', ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga', 'lbfgs'])
        
        params = {
            'alpha': trial.suggest_float('alpha', 1e-4, 100.0, log=True),
            'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
            'copy_X': trial.suggest_categorical('copy_X', [True]),
            'tol': trial.suggest_float('tol', 1e-6, 1e-2, log=True),
            'solver': trial.suggest_categorical('solver', ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga', 'lbfgs']),
            'positive': trial.suggest_categorical('positive', [True]) if solver == 'lbfgs' else False,
            'random_state': trial.suggest_categorical('random_state', [42]) if solver in ['sag', 'saga'] else None,
            'max_iter': trial.suggest_int('max_iter', 500, 5000) if solver in ['sag', 'saga', 'sparse_cg', 'lsqr'] else None,
        }
        
        return params


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