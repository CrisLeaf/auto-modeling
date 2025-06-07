from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

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

            model = LinearRegression(fit_intercept=fit_intercept, tol=tol)
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