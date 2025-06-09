import importlib.util
import pytest
from sklearn.datasets import make_regression

from automodeling.linear_model import AutoRidge


plotly_installed = importlib.util.find_spec("plotly") is not None


@pytest.mark.skipif(not plotly_installed, reason="plotly is not installed")
def test_study_visualizations():
    try:
        import optuna.visualization as vis
    except ImportError:
        pytest.skip('optuna.visualization could not be imported')
    
    X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, n_targets=1, random_state=42)
    
    model = AutoRidge(
        auto_scoring='neg_mean_squared_error',
        auto_direction='minimize',
        auto_n_trials=100,
        auto_verbose=True,
        auto_use_scaler=True
    )
    model.search_fit(X, y)
    
    fig1 = vis.plot_optimization_history(model.study_)
    fig2 = vis.plot_param_importances(model.study_)
    fig3 = vis.plot_slice(model.study_)
    
    assert fig1 is not None, 'plot_optimization_history returned None'
    assert fig2 is not None, 'plot_param_importances returned None'
    assert fig3 is not None, 'plot_slice returned None'
    
    assert fig1.__class__.__name__ ==  "Figure", "Fig1 Expected a Plotly Figure object"
    assert fig2.__class__.__name__ ==  "Figure", "Fig2 Expected a Plotly Figure object"
    assert fig3.__class__.__name__ ==  "Figure", "Fig3 Expected a Plotly Figure object"