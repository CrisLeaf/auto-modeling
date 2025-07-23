from sklearn.datasets import make_regression, make_classification
from sklearn.metrics import mean_squared_error, accuracy_score


def _get_regression_X_y():
    return make_regression(n_samples=1000, n_features=10, n_informative=5, n_targets=1, random_state=42)

def _get_classification_X_y():
    return make_classification(n_samples=1000, n_features=10, n_informative=5, n_classes=2, random_state=42)

def model_basics_test(model, any_parameter_name, max_mse=10.0):
    X, y = _get_regression_X_y()
    
    model.search_fit(X, y)
    
    assert model._is_searched
    assert model.pipeline_ is not None, 'pipeline_ was not created after fitting'
    assert model.study_ is not None, 'study_ was not initialized'
    
    preds = model.predict(X)
    assert preds.shape == y.shape, 'Predictions shape is not correct'
    
    params = model.get_params()
    assert any_parameter_name in params, f'get_params does not return {any_parameter_name} parameter'
    
    mse = mean_squared_error(y, preds)
    assert mse < max_mse, f'MSE is too high: {mse}'
    
    print('Best trial params:', model.study_.best_params)

def model_basics_test_binary_classification(model, any_parameter_name, min_acc=0.8):
    X, y = _get_classification_X_y()
    
    model.search_fit(X, y)
    
    assert model._is_searched
    assert model.pipeline_ is not None, 'pipeline_ was not created after fitting'
    assert model.study_ is not None, 'study_ was not initialized'
    
    preds = model.predict(X)
    assert preds.shape == y.shape, 'Predictions shape is not correct'
    
    params = model.get_params()
    assert any_parameter_name in params, f'get_params does not return {any_parameter_name} parameter'
    
    acc = accuracy_score(y, preds)
    assert acc > min_acc, f'Accuracy is too low: {acc}'
    
    print('Best trial params:', model.study_.best_params)