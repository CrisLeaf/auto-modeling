from automodeling.linear_model import AutoLinearRegression, AutoRidge
from utils import model_basics_test


def test_auto_linear_regression():
    model = AutoLinearRegression(
        auto_scoring='neg_mean_squared_error',
        auto_direction='minimize',
        auto_n_trials=100,
        auto_timeout=60*2,
        auto_verbose=True,
        auto_use_scaler=True
    )
    
    model_basics_test(model, 'fit_intercept')

def test_auto_ridge():
    model = AutoRidge(
        auto_scoring='neg_mean_squared_error',
        auto_direction='minimize',
        auto_n_trials=100,
        auto_timeout=60*2,
        auto_verbose=True,
        auto_use_scaler=True
    )
    
    model_basics_test(model, 'alpha')