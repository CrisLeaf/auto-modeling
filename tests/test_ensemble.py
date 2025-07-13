from automodeling.ensemble import AutoRandomForestRegressor, AutoGradientBoostingRegressor
from utils import model_basics_test


def test_auto_random_forest_regressor():
    model = AutoRandomForestRegressor(
        auto_scoring='neg_mean_squared_error',
        auto_direction='minimize',
        auto_n_trials=100,
        auto_timeout=60*2,
        auto_verbose=True,
        auto_use_scaler=False
    )
    
    model_basics_test(model, 'n_estimators', max_mse=400.0)

def test_auto_gradient_boosting_regressor():
    model = AutoGradientBoostingRegressor(
        auto_scoring='neg_mean_squared_error',
        auto_direction='minimize',
        auto_n_trials=100,
        auto_timeout=60*2,
        auto_verbose=True,
        auto_use_scaler=False
    )
    
    model_basics_test(model, 'n_estimators', max_mse=200.0)
