# !pip install pytest
#!pip install mlflow
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '../notebooks'))) 
#'/Workspace/Users/sagarbansal719@gmail.com/ML_Clf_Model/notebooks')

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock
from run_model_inference_py import ModelLoader, WineDataProcessor, WineQualityPredictor

# Mock MLflow model for testing
class MockModel:
    def predict(self, X):
        return np.random.rand(len(X))  # Simulated probability outputs

@pytest.fixture
def mock_model_loader():
    """Fixture to mock the model loading."""
    model_loader = ModelLoader("wine_quality")
    model_loader.model = MockModel()  # Use mock model instead of real MLflow model
    return model_loader

@pytest.fixture
def mock_data_processor():
    """Fixture to mock data processing."""
    feature_columns = ['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
                       'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density',
                       'pH', 'sulphates', 'alcohol', 'is_red']
    processor = WineDataProcessor(feature_columns)
    
    # Mocking data instead of loading from CSV
    processor.data = pd.DataFrame(np.random.rand(5, len(feature_columns)), columns=feature_columns)
    return processor

def test_model_loading(mock_model_loader):
    """Test if the model loads successfully."""
    assert mock_model_loader.get_model() is not None, "Model should be loaded"

def test_prediction_output(mock_model_loader, mock_data_processor):
    """Test if predictions have the correct shape and type."""
    predictor = WineQualityPredictor(mock_model_loader, mock_data_processor)
    predictions = predictor.predict()

    assert isinstance(predictions, np.ndarray), "Predictions should be a numpy array"
    assert predictions.shape == (5,), "Prediction shape should match the number of input samples"
    assert np.all((predictions >= 0) & (predictions <= 1)), "Predictions should be between 0 and 1"


#To test in UI
# if __name__ == "__main__":
#     pytest.main()
