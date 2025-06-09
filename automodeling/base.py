from abc import ABC, abstractmethod
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

import optuna

class AutoModelBase(ABC):
    def __init__(self, auto_scoring=None, auto_direction='minimize', auto_timeout=60, auto_n_trials=None, auto_verbose=False, auto_use_scaler=False):
        self.auto_scoring = auto_scoring
        self.auto_direction = auto_direction
        self.auto_timeout = auto_timeout
        self.auto_n_trials = auto_n_trials
        self.auto_verbose = auto_verbose
        self.auto_use_scaler = auto_use_scaler
        self.best_params_ = None
        self.pipeline_ = None
        self._study = None
        self._is_searched = False
    
    @abstractmethod
    def _get_model_params(self):
        """Returns a dictionary of model parameters."""
        pass
    
    @abstractmethod
    def _build_model(self, params):
        """Receives a dictionary of model parameters and returns a model."""
        pass
    
    @abstractmethod
    def _get_search_space(self, trial):
        """Defines the search space for hyperparameters and return a dictionary."""
        pass
    
    @property
    def study_(self):
        if not self._is_searched or self._study is None:
            raise ValueError('The study has not been executed yet. Please call search_fit first.')
        
        return self._study
    
    def search(self, X, y, cv=5):
        def objective(trial):
            params = self._get_search_space(trial)
            model = self._build_model(params)
            
            if self.auto_use_scaler:
                pipeline = make_pipeline(StandardScaler(), model)
            else:
                pipeline = model
            
            score = -cross_val_score(pipeline, X, y, cv=cv, scoring=self.auto_scoring).mean()
            
            return score
        
        self._study = optuna.create_study(direction=self.auto_direction)
        self._study.optimize(
            objective,
            timeout=self.auto_timeout,
            n_trials=self.auto_n_trials,
            show_progress_bar=self.auto_verbose
        )
        
        self.best_params_ = self._study.best_params
        self._is_searched = True
        
        for param, value in self.best_params_.items():
            setattr(self, param, value)
        
    def fit(self, X, y):
        params = self._get_model_params()
        model = self._build_model(params)
        
        if self.auto_use_scaler:
            self.pipeline_ = make_pipeline(StandardScaler(), model)
        else:
            self.pipeline_ = model
        
        self.pipeline_.fit(X, y)
        
        return self
    
    def predict(self, X):
        return self.pipeline_.predict(X)
    
    def search_fit(self, X, y, cv=5):
        self.search(X, y, cv=cv)
        self.fit(X, y)
        
        return self