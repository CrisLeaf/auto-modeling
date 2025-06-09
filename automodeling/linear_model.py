from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, RegressorMixin

from automodeling.base import AutoModelBase

import logging
import optuna

logging.getLogger("optuna").setLevel(logging.WARNING)


class AutoLinearRegression(LinearRegression):
    def __init__(self, scoring=None, timeout=60, n_trials=None, verbose=False, **kwargs):
        super().__init__(**kwargs)
        self.scoring = scoring
        self.timeout = timeout
        self.n_trials = n_trials
        self.verbose = verbose
        self.best_params_ = None
        self.study_ = None
        self._is_searched = False
    
    def search(self, X, y, cv=5):
        def objective(trial):
            fit_intercept = trial.suggest_categorical('fit_intercept', [True, False])
            tol = trial.suggest_float('tol', 1e-8, 1e-2, log=True)

            model = LinearRegression(
                fit_intercept=fit_intercept
            )
            pipeline = make_pipeline(StandardScaler(), model)
            score = -cross_val_score(pipeline, X, y, cv=cv, scoring=self.scoring).mean()
            
            return score
        
        self.study_ = optuna.create_study(direction='minimize')
        self.study_.optimize(
            objective,
            timeout=self.timeout,
            n_trials=self.n_trials,
            show_progress_bar=self.verbose
        )
        
        self.best_params_ = self.study_.best_params
        self._is_searched = True
        
        for param, value in self.best_params_.items():
            setattr(self, param, value)
        
    def fit(self, X, y):
        return super().fit(X, y)
    
    def search_fit(self, X, y, cv=5):
        self.search(X, y, cv=cv)
        self.fit(X, y)

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
            'copy_X': True,
            'tol': trial.suggest_float('tol', 1e-6, 1e-2, log=True),
            'solver': solver,
            'positive': solver == 'lbfgs',
            'random_state': 42 if solver in ['sag', 'saga'] else None,
            'max_iter': trial.suggest_int('max_iter', 500, 5000) if solver in ['sag', 'saga', 'sparse_cg', 'lsqr'] else None,
        }
        
        return params