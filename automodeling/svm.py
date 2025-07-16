from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.svm import SVR

from automodeling.base import AutoModelBase

import logging

logging.getLogger("optuna").setLevel(logging.WARNING)


class AutoSVR(AutoModelBase, BaseEstimator, RegressorMixin):
    def __init__(
            self,
            kernel='rbf',
            degree=3,
            gamma='scale',
            coef0=0.0,
            tol=0.001,
            C=1.0,
            epsilon=0.1,
            shrinking=True,
            cache_size=200,
            verbose=False,
            max_iter=-1,
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
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.C = C
        self.epsilon = epsilon
        self.shrinking = shrinking
        self.cache_size = cache_size
        self.verbose = verbose
        self.max_iter = max_iter
        
    def _get_model_params(self):
        return {
            'kernel': self.kernel,
            'degree': self.degree,
            'gamma': self.gamma,
            'coef0': self.coef0,
            'tol': self.tol,
            'C': self.C,
            'epsilon': self.epsilon,
            'shrinking': self.shrinking,
            'cache_size': self.cache_size,
            'verbose': self.verbose,
            'max_iter': self.max_iter
        }
        
    def _build_model(self, params):
        return SVR(**params)
    
    def _get_search_space(self, trial):
        params = {
            'kernel': trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
            'degree': trial.suggest_int('degree', 2, 5) if trial.suggest_categorical('use_degree', [True, False]) else 3,
            'gamma': trial.suggest_float('gamma_value', 1e-4, 10.0, log=True) if trial.suggest_categorical('gamma_float', [True, False]) else trial.suggest_categorical('gamma', ['scale', 'auto']),
            'coef0': trial.suggest_float('coef0', 0.0, 1.0),
            'tol': trial.suggest_float('tol', 1e-4, 1e-2, log=True),
            'C': trial.suggest_float('C', 0.1, 100.0, log=True),
            'epsilon': trial.suggest_float('epsilon', 0.01, 1.0),
            'shrinking': trial.suggest_categorical('shrinking', [True, False]),
            'cache_size': 200,
            'verbose': False,
            'max_iter': -1
        }
    
        return params