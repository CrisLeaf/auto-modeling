from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.svm import SVR, SVC

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
            'max_iter': trial.suggest_int('max_iter', 500, 5000)
        }
    
        return params

class AutoSVC(AutoModelBase, BaseEstimator, ClassifierMixin):
    def __init__(
            self,
            C=1.0,
            kernel='rbf',
            degree=3,
            gamma='scale',
            coef0=0.0,
            shrinking=True,
            probability=False,
            tol=0.001,
            cache_size=200,
            class_weight=None,
            verbose=False,
            max_iter=-1,
            decision_function_shape='ovr',
            break_ties=False,
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
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.shrinking = shrinking
        self.probability = probability
        self.tol = tol
        self.cache_size = cache_size
        self.class_weight = class_weight
        self.verbose = verbose
        self.max_iter = max_iter
        self.decision_function_shape = decision_function_shape
        self.break_ties = break_ties
        self.random_state = random_state
        
    def _get_model_params(self):
        return {
            'C': self.C,
            'kernel': self.kernel,
            'degree': self.degree,
            'gamma': self.gamma,
            'coef0': self.coef0,
            'shrinking': self.shrinking,
            'probability': self.probability,
            'tol': self.tol,
            'cache_size': self.cache_size,
            'class_weight': self.class_weight,
            'verbose': self.verbose,
            'max_iter': self.max_iter,
            'decision_function_shape': self.decision_function_shape,
            'break_ties': self.break_ties,
            'random_state': self.random_state,
        }
        
    def _build_model(self, params):
        return SVC(**params)
    
    def _get_search_space(self, trial):
        decision_function_shape = trial.suggest_categorical('decision_function_shape', ['ovo', 'ovr'])
        
        if decision_function_shape == 'ovr':
            break_ties = trial.suggest_categorical('break_ties', [True, False])
        else:
            break_ties = False
        
        params = {
            'C': trial.suggest_float('C', 0.1, 100.0, log=True),
            'kernel': trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
            'degree': trial.suggest_int('degree', 2, 5),
            'gamma': trial.suggest_float('gamma_value', 1e-4, 10.0, log=True) if trial.suggest_categorical('gamma_float', [True, False]) else trial.suggest_categorical('gamma', ['scale', 'auto']),
            'coef0': trial.suggest_float('coef0', 0.0, 1.0),
            'shrinking': trial.suggest_categorical('shrinking', [True, False]),
            'probability': self.probability,
            'tol': trial.suggest_float('tol', 1e-4, 1e-2, log=True),
            'cache_size': self.cache_size,
            'class_weight': trial.suggest_categorical('class_weight', ['balanced', None]),
            'verbose': self.verbose,
            'max_iter': trial.suggest_int('max_iter', 100_000, 500_000),
            'decision_function_shape': decision_function_shape,
            'break_ties': break_ties,
            'random_state': trial.suggest_categorical('random_state', [42])
        }
        
        return params