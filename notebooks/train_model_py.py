#!pip install mlflow

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
import os
import sys

# Load Databricks credentials from environment variables
DATABRICKS_HOST = os.getenv("DATABRICKS_HOST")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")

#Extracting dynamic code configurations
try:
    if (len(sys.argv) > 1) &('-f' not in sys.argv[1]):
        catalog = sys.argv[1] if (len(sys.argv) > 1) &('-f' not in sys.argv[1]) else "workspace"
        schema = sys.argv[2] if (len(sys.argv) > 1) &('-f' not in sys.argv[2]) else "wine_quality_data"
except:
    catalog = "workspace"
    schema = "wine_quality_data"

#For UI Run:
# catalog = "workspace"
# schema = "wine_quality_data"
print(catalog, schema)

#Create a SparkSession and set it as the default context
spark = SparkSession.builder.config("spark.databricks.service.client.enabled", "true").config("spark.databricks.service.token", DATABRICKS_TOKEN).config("spark.databricks.unityCatalog.enabled", "true").getOrCreate()

#mlflow uri setup
mlflow.set_registry_uri("databricks-uc")
mlflow.set_tracking_uri("databricks")
#mlflow.set_experiment(experiment_id = "869fe2efae694fcba0e661edc32a1727")
#mlflow.set_experiment(experiment_id = "869fe2efae694fcba0e661edc32a1727")

# 1. Data Processing Class
class WineDataProcessor:
    def __init__(self):
        self.data = None

    def load_data(self):
        """Loads wine datasets and preprocesses them."""

        #Debug the issue - print all databases and corresponding table names
        db = spark.catalog.listDatabases()
        for database1 in db:
            print("database_name:", database1.name)
            spark.catalog.setCurrentDatabase(database1.name)
            tables = spark.catalog.listTables()
            for table in tables:
                print("table_name:", table.name)

        #Debug the issue - print all catalogs and schemas
        df_schema = spark.sql("SHOW CURRENT SCHEMA").toPandas()
        df_catalog = spark.sql("SHOW CATALOGS").toPandas()
        print("Current Schema:", df_schema['catalog'][0], df_schema['namespace'][0], "Catalogs:", list(df_catalog['catalog'])  )
        
        #Catalog - throws error here
        catalog_query = f"USE CATALOG {catalog};"
        schema_query = f"USE SCHEMA {schema};"
        spark.sql(catalog_query)
        spark.sql(schema_query)

        white_wine = spark.read.format("delta").table(f"{catalog}.{schema}.white_wine_training_data").toPandas()
        red_wine = spark.read.format("delta").table(f"{catalog}.{schema}.red_wine_training_data").toPandas()

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

        mlflow.pyfunc.log_model("wine_quality_model", python_model=wrappedModel, conda_env=conda_env, signature=signature)

# 4. MLflow Experiment Class
class MLflowExperiment:
    def __init__(self, model_name="wine_quality_model"):
        self.model_name = model_name

    def run_experiment(self, model, X_train, X_test, y_train, y_test):
        """Runs an MLflow experiment to log parameters and metrics."""
        #Explicit set and verify the mlflow run
        mlflow.set_tracking_uri("databricks")
        print("Tracking UI:", mlflow.get_tracking_uri())
        
        with mlflow.start_run(run_name='wine_model_training_run'):
            model.train(X_train, y_train)
            auc_score = model.evaluate(X_test, y_test)

            mlflow.log_param('n_estimators', model.model.n_estimators)
            mlflow.log_metric('auc', auc_score)
            model.log_model(X_train)

            return mlflow.search_runs(filter_string='tags.mlflow.runName = "wine_model_training_run"').iloc[0].run_id

    def register_model(self, run_id):
        """Registers the trained model in MLflow."""
        model_version = mlflow.register_model(f"runs:/{run_id}/wine_quality_model", self.model_name)
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

    #To review prediction data
    print(y_test.head())
    
    # Model Training
    wine_model = WineQualityModel()
    experiment = MLflowExperiment()

    # Run experiment & register model
    run_id = experiment.run_experiment(wine_model, X_train, X_test, y_train, y_test)
    model_version = experiment.register_model(run_id)
    
    # Feature Importance
    feature_importance = FeatureImportance.get_importance(wine_model, X_train)
    print(feature_importance)
