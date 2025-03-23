# !pip install mlflow

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from mlflow.models.signature import infer_signature
from mlflow.utils.environment import _mlflow_conda_env
import cloudpickle
import time
from pyspark.sql.session import SparkSession
spark = SparkSession.builder.getOrCreate()

# Load Databricks credentials from environment variables
DATABRICKS_HOST = os.getenv("DATABRICKS_HOST")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")

#Create a SparkSession and set it as the default context
spark = SparkSession.builder.config("spark.databricks.service.client.enabled", "true").config("spark.databricks.service.token", DATABRICKS_TOKEN).config("spark.databricks.unityCatalog.enabled", "true").getOrCreate()

#Spark version check
print(spark.version)

# Set the MLflow model registry URI
#spark.conf.set("spark.mlflow.modelRegistryUri", "databricks")
mlflow.set_registry_uri("databricks-uc")
mlflow.set_tracking_uri("databricks")
mlflow.set_experiment("/Users/sagarbansal719@gmail.com/Wine_Quality_Prediction_Model/notebooks/train_model_py.py") 

# 1. Model Loader Class
class ModelLoader:
    def __init__(self, model_name, model_version=None):
        """
        Initializes the ModelLoader class.

        :param model_name: Name of the model in MLflow Registry.
        :param model_version: (Optional) Specific version to load. Defaults to latest version.
        """
        self.model_name = model_name
        self.model_version = model_version
        self.model = None

    def load_model(self):
        """Loads the model from MLflow registry."""
        model_uri = f"models:/{self.model_name}/{self.model_version}" if self.model_version else f"models:/{self.model_name}@production"
        self.model = mlflow.pyfunc.load_model(model_uri)
        print(f"Model '{self.model_name}' loaded successfully.")

    def get_model(self):
        """Returns the loaded model instance."""
        if self.model is None:
            raise ValueError("Model is not loaded. Call `load_model()` first.")
        return self.model

# 2. Data Preprocessing Class
class WineDataProcessor:
    def __init__(self, feature_columns):
        self.data = None
        self.feature_columns = feature_columns

    def load_data(self):
        """Loads red and white wine datasets and preprocesses them."""
        data = spark.read.table("wine_quality_data.wine_quality_inference_data").toPandas()
        
        # Check for missing columns
        missing_cols = [col for col in self.feature_columns if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required features: {missing_cols}")

        self.data = data
        return self.data
    
# 3. Inference Class
class WineQualityPredictor:
    def __init__(self, model_loader, preprocessor):
        """
        Initializes the predictor with a loaded model and preprocessor.

        :param model_loader: Instance of ModelLoader.
        :param preprocessor: Instance of DataPreprocessor.
        """
        self.model = model_loader.get_model()
        self.preprocessor = preprocessor

    def predict(self):
        """Runs inference on input data."""
        processed_data = self.preprocessor.load_data()
        predictions = self.model.predict(processed_data)
        
        predictions_df = pd.DataFrame(predictions, columns=["prediction"])
        predictions_df
        results_df = pd.concat([processed_data, predictions_df], axis =1)
        results_df['prediction'] = results_df['prediction'].map(lambda x: 1 if x > 0.5 else 0)
        return results_df

# Main Execution
if __name__ == "__main__":
    # Define the model name (ensure this matches the registered model in MLflow)
    MODEL_NAME = "wine_quality"

    # Load model
    model_loader = ModelLoader(MODEL_NAME)
    model_loader.load_model()

    # Define feature columns (must match those used during training)
    FEATURE_COLUMNS = ['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
       'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density',
       'pH', 'sulphates', 'alcohol', 'is_red']

    # Initialize preprocessor
    preprocessor = WineDataProcessor(FEATURE_COLUMNS)

    #To review the data
    #print(preprocessor.load_data())

    # Initialize predictor
    predictor = WineQualityPredictor(model_loader, preprocessor)

    # Run inference
    results_df = predictor.predict()
    
    #View the results sample
    print("here are the results:")
    print(results_df.head())

    # Save results to Delta table in Unity Catalog
    spark.createDataFrame(results_df).write.format("delta").mode("overwrite").saveAsTable(f"wine_quality_data.wine_quality_predictions")

    # Print results
    for i, pred in enumerate(results_df['prediction']):
        print(f"Sample {i+1} → High Quality Probability: {pred:.2f}")
