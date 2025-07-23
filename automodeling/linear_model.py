from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression

from automodeling.base import AutoModelBase

import logging

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
            'max_iter': trial.suggest_int('max_iter', 1000, 5000) if solver in ['sag', 'saga', 'sparse_cg', 'lsqr'] else None,
        }
        
        return params


class AutoLogisticRegression(AutoModelBase, BaseEstimator, ClassifierMixin):
    def __init__(
            self,
            penalty='l2',
            dual=False,
            tol=1e-4,
            C=1.0,
            fit_intercept=True,
            intercept_scaling=1,
            class_weight=None,
            random_state=None,
            solver='lbfgs',
            max_iter=100,
            multi_class='auto',
            verbose=0,
            warm_start=False,
            n_jobs=None,
            l1_ratio=None,
            auto_scoring=None,
            auto_direction='maximize',
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
        self.penalty = penalty
        self.dual = dual
        self.tol = tol
        self.C = C
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.class_weight = class_weight
        self.random_state = random_state
        self.solver = solver
        self.max_iter = max_iter
        self.multi_class = multi_class
        self.verbose= verbose
        self.warm_start = warm_start
        self.n_jobs = n_jobs
        self.l1_ratio = l1_ratio
        
    def _get_model_params(self):
        return {
            'penalty': self.penalty,
            'dual': self.dual,
            'tol': self.tol,
            'C': self.C,
            'fit_intercept': self.fit_intercept,
            'intercept_scaling': self.intercept_scaling,
            'class_weight': self.class_weight,
            'random_state': self.random_state,
            'solver': self.solver,
            'max_iter': self.max_iter,
            'multi_class': self.multi_class,
            'verbose': self.verbose,
            'warm_start': self.warm_start,
            'n_jobs': self.n_jobs,
            'l1_ratio': self.l1_ratio
        }
        
    def _build_model(self, params):
        return LogisticRegression(**params)
    
    def _get_search_space(self, trial):
        params = {
            'penalty': self.penalty,
            'dual': self.dual,
            'tol': trial.suggest_float('tol', 1e-6, 1e-2, log=True),
            'C': trial.suggest_float('C', 1e-4, 100.0, log=True),
            'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
            'intercept_scaling': trial.suggest_float('intercept_scaling', 1e-4, 100.0, log=True),
            'class_weight': trial.suggest_categorical('class_weight', ['balanced', None]),
            'random_state': trial.suggest_categorical('random_state', [42]),
            'solver': trial.suggest_categorical('solver', ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']),
            'max_iter': trial.suggest_int('max_iter', 1000, 5000),
            'multi_class': self.multi_class,
            'verbose': self.verbose,
            'warm_start': self.warm_start,
            'n_jobs': self.n_jobs,
            'l1_ratio': self.l1_ratio
        }
        
        return params
        