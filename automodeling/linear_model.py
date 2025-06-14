from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, RegressorMixin

from automodeling.base import AutoModelBase

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
        
        if solver == 'lbfgs':
            positive = True
        else:
            positive = False
        
        params = {
            'alpha': trial.suggest_float('alpha', 1e-4, 100.0, log=True),
            'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
            'copy_X': True,
            'tol': trial.suggest_float('tol', 1e-6, 1e-2, log=True),
            'solver': solver,
            'positive': positive,
            'random_state': 42 if solver in ['sag', 'saga'] else None,
            'max_iter': trial.suggest_int('max_iter', 500, 5000) if solver in ['sag', 'saga', 'sparse_cg', 'lsqr'] else None,
        }
        
        return params