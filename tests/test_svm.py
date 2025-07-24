from automodeling.svm import AutoSVR
from utils import model_basics_test

def test_auto_random_forest_regressor():
    model = AutoSVR(
        auto_scoring='neg_mean_squared_error',
        auto_direction='minimize',
        auto_n_trials=100,
        auto_timeout=60*2,
        auto_verbose=True,
        auto_use_scaler=False
    )
    
    model_basics_test(model, 'kernel', max_mse=400.0)