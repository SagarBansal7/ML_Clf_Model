# !pip install pytest
# !pip install mlflow
import sys
sys.path.insert(0, '/Workspace/Users/sagarbansal719@gmail.com/ML_Clf_Model/notebooks')

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
from sklearn.ensemble import RandomForestClassifier
from train_model_py import WineDataProcessor, WineQualityModel, MLflowExperiment, FeatureImportance
import mlflow
import mlflow.pyfunc
import mlflow.sklearn

# Mock Data
@pytest.fixture
def mock_data():
    """Fixture to provide mock data for training/testing."""
    processor = WineDataProcessor()
    data = processor.load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = processor.split_data()
    return X_train, X_val, X_test, y_train, y_val, y_test

@pytest.fixture
def mock_model():
    """Fixture to provide a mock model for testing."""
    model = WineQualityModel(RandomForestClassifier(n_estimators=10, random_state=123))
    return model

def test_data_loading(mock_data):
    """Test that the data loads and splits correctly."""
    X_train, X_val, X_test, y_train, y_val, y_test = mock_data
    
    assert isinstance(X_train, pd.DataFrame), "X_train should be a pandas DataFrame"
    assert X_train.shape[0] > 0, "Training data should have samples"
    assert len(y_train) == X_train.shape[0], "X_train and y_train should have the same number of samples"

def test_model_training(mock_data, mock_model):
    """Test the model training process."""
    X_train, X_val, X_test, y_train, y_val, y_test = mock_data
    mock_model.train(X_train, y_train)
    
    # Ensure the model is trained (it should have learned parameters)
    assert hasattr(mock_model.model, 'n_estimators'), "Model should have a trained classifier"
    assert mock_model.model.n_estimators == 10, "RandomForestClassifier should have 10 estimators"

def test_model_evaluation(mock_data, mock_model):
    """Test the model evaluation process."""
    X_train, X_val, X_test, y_train, y_val, y_test = mock_data
    mock_model.train(X_train, y_train)
    
    auc_score = mock_model.evaluate(X_test, y_test)
    
    assert isinstance(auc_score, float), "AUC score should be a float"
    assert 0 <= auc_score <= 1, "AUC score should be between 0 and 1"

def test_mlflow_logging(mock_data, mock_model):
    """Test that the model is logged correctly to MLflow."""
    X_train, X_val, X_test, y_train, y_val, y_test = mock_data
    experiment = MLflowExperiment(model_name="wine_quality_test")
    
    # Mock MLflow's start_run to avoid actual logging
    with pytest.monkeypatch.context() as mp:
        mp.setattr(mlflow, 'start_run', MagicMock())
        
        # Run experiment & log model
        run_id = experiment.run_experiment(mock_model, X_train, X_test, y_train, y_test)
        
        # Check that MLflow's start_run was called
        mlflow.start_run.assert_called_once()
        assert run_id is not None, "Run ID should be returned from the experiment"

def test_feature_importance(mock_data, mock_model):
    """Test feature importance extraction."""
    X_train, X_val, X_test, y_train, y_val, y_test = mock_data
    mock_model.train(X_train, y_train)
    
    feature_importance = FeatureImportance.get_importance(mock_model, X_train)
    
    assert isinstance(feature_importance, pd.DataFrame), "Feature importance should be a pandas DataFrame"
    assert 'importance' in feature_importance.columns, "Feature importance DataFrame should contain 'importance' column"
    assert feature_importance.shape[0] == X_train.shape[1], "Feature importance should match the number of features"

