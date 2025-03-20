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

#mlflow uri setup
mlflow.set_registry_uri("databricks-uc")
mlflow.set_tracking_uri("databricks")
mlflow.set_experiment("/Users/sagarbansal719@gmail.com/ML_Clf_Model/notebooks/train_model_py.py") 

# 1. Data Processing Class
class WineDataProcessor:
    def __init__(self):
        self.data = None

    def load_data(self):
        """Loads wine datasets and preprocesses them."""
        spark.sql("USE catalog workspace")
        spark.sql("USE schema default")
        df_schema = spark.sql("SHOW CURRENT SCHEMA").toPandas()
        print("Current Schema:", df_schema['catalog'][0], df_schema['namespace'][0] )
        white_wine = spark.read.format("delta").table("white_wine_training_data").toPandas()
        red_wine = spark.read.format("delta").table("red_wine_training_data").toPandas()

        red_wine['is_red'] = 1
        white_wine['is_red'] = 0
        data = pd.concat([red_wine, white_wine], axis=0)

        # Clean column names
        data.rename(columns=lambda x: x.replace(' ', '_'), inplace=True)

        # Convert quality into a binary classification (high quality or not)
        data['quality'] = (data.quality >= 7).astype(int)

        self.data = data
        return self.data

    def split_data(self, test_size=0.2, val_size=0.2):
        """Splits the dataset into training, validation, and test sets."""
        X = self.data.drop(["quality"], axis=1)
        y = self.data.quality

        X_train, X_rem, y_train, y_rem = train_test_split(X, y, train_size=1-test_size-val_size, random_state=123)
        X_val, X_test, y_val, y_test = train_test_split(X_rem, y_rem, test_size=val_size/(test_size+val_size), random_state=123)

        return X_train, X_val, X_test, y_train, y_val, y_test

# 2. Model Wrapper Class
class SklearnModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input):
        return self.model.predict_proba(model_input)[:,1]

# 3. Machine Learning Model Class
class WineQualityModel:
    def __init__(self, model=RandomForestClassifier(n_estimators=10, random_state=123)):
        self.model = model

    def train(self, X_train, y_train):
        """Trains the model."""
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        """Evaluates the model using ROC AUC score."""
        predictions = self.model.predict_proba(X_test)[:,1]
        auc_score = roc_auc_score(y_test, predictions)
        return auc_score

    def log_model(self, X_train):
        """Logs the model using MLflow."""
        wrappedModel = SklearnModelWrapper(self.model)
        signature = infer_signature(X_train, wrappedModel.predict(None, X_train))

        conda_env = _mlflow_conda_env(
            additional_conda_deps=None,
            additional_pip_deps=["cloudpickle=={}".format(cloudpickle.__version__), "scikit-learn=={}".format(sklearn.__version__)],
            additional_conda_channels=None,
        )

        mlflow.pyfunc.log_model("random_forest_model", python_model=wrappedModel, conda_env=conda_env, signature=signature)

# 4. MLflow Experiment Class
class MLflowExperiment:
    def __init__(self, model_name="wine_quality"):
        self.model_name = model_name

    def run_experiment(self, model, X_train, X_test, y_train, y_test):
        """Runs an MLflow experiment to log parameters and metrics."""
        with mlflow.start_run(run_name='untuned_random_forest'):
            model.train(X_train, y_train)
            auc_score = model.evaluate(X_test, y_test)

            mlflow.log_param('n_estimators', model.model.n_estimators)
            mlflow.log_metric('auc', auc_score)
            model.log_model(X_train)

            return mlflow.search_runs(filter_string='tags.mlflow.runName = "untuned_random_forest"').iloc[0].run_id

    def register_model(self, run_id):
        """Registers the trained model in MLflow."""
        model_version = mlflow.register_model(f"runs:/{run_id}/random_forest_model", self.model_name)
        client = mlflow.MlflowClient()
        client.set_registered_model_alias(self.model_name, "production", version=model_version.version)
        time.sleep(15)  # Wait for registration to complete
        return model_version

# 5. Feature Importance Class
class FeatureImportance:
    @staticmethod
    def get_importance(model, X_train):
        """Retrieves and sorts feature importance values."""
        feature_importances = pd.DataFrame(model.model.feature_importances_, index=X_train.columns, columns=['importance'])
        return feature_importances.sort_values('importance', ascending=False)

# Main Script
if __name__ == "__main__":
    # Data Processing
    processor = WineDataProcessor()
    data = processor.load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = processor.split_data()

    # Model Training
    wine_model = WineQualityModel()
    experiment = MLflowExperiment()

    # Run experiment & register model
    run_id = experiment.run_experiment(wine_model, X_train, X_test, y_train, y_test)
    model_version = experiment.register_model(run_id)
    

    # Feature Importance
    feature_importance = FeatureImportance.get_importance(wine_model, X_train)
    print(feature_importance)
