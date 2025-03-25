# Wine Quality Prediction Documentation

## Repository Overview

**ML_Clf_Model** is a machine learning project designed for training and deploying a Wine Quality Prediction model using Databricks. The repository includes scripts for data processing, model training, inference, CI/CD workflows, and testing.

---

## Repository Structure

```
ML_Clf_Model/
│── .github/
│   └── ModelWorkflow.yml  # GitHub Actions workflow for CI/CD
│── cli_tool/
│   ├── cli.py  # CLI tool for deploying Databricks jobs
│   ├── databricks_jobs.py  # Manages job deployment in Databricks
│── notebooks/
│   ├── train_model_py.py  # Script for training the model
│   ├── run_model_inference_py.py  # Script for running inference
│── tests/
│   ├── test_training.py  # Unit tests for model training
│   ├── test_inference.py  # Unit tests for inference
│── Readme.md  # Project documentation
│── requirements.txt  # List of dependencies
```

---

## CI/CD Workflow

The repository uses **GitHub Actions** for automation. The workflow is defined in `.github/ModelWorkflow.yml` and consists of the following jobs:

### 1. **Dev**
- Runs on every push to `dev` branch.
- Sets up Python, installs dependencies, runs tests and executes the CLI tool to deploy jobs to Databricks (development environment).

### 2. **Stage**
- Runs on every push to `stage` branch.
- Sets up Python, installs dependencies, runs tests and executes the CLI tool to deploy jobs to Databricks (stage environment).

### 3. **Main**
- Runs on every push to `main` branch.
- Sets up Python, installs dependencies, runs tests and executes the CLI tool to deploy jobs to Databricks (production environment).

---

## CLI Tool

### **cli_tool/cli.py**
A command-line interface to deploy jobs to Databricks.

**Commands:**
- `deploy`: Deploys training and inference jobs to Databricks.

**Usage:**
```sh
python cli_tool/cli.py deploy
```

---

## Model Training

### **notebooks/train_model_py.py**
This script:
- Loads and preprocesses wine quality data.
- Trains a **Random Forest Classifier**.
- Logs the model using **MLflow**.
- Registers the model in MLflow.

**Key Classes:**
- `WineDataProcessor`: Handles data loading and splitting.
- `WineQualityModel`: Defines the classifier and evaluation methods.
- `MLflowExperiment`: Manages experiment tracking and model registration.
- `FeatureImportance`: Extracts feature importance from the trained model.

---

## Model Inference

### **notebooks/run_model_inference_py.py**
This script:
- Loads the trained model from MLflow.
- Performs inference on new data.
- Saves results to a **Delta table** in Unity Catalog.

**Key Classes:**
- `ModelLoader`: Loads models from MLflow.
- `WineDataProcessor`: Prepares inference data.
- `WineQualityPredictor`: Runs inference using the trained model.
- `DeltaTableSaver`: Saves results to a Delta table.

---

## Databricks Job Management

### **cli_tool/databricks_jobs.py**
Handles Databricks job creation via API.

**Key Methods:**
- `create_job`: Creates a Databricks job.
- `create_training_job`: Deploys the training job (runs every 30 days).
- `create_inference_job`: Deploys the inference job (runs daily).

---

## Testing

### **tests/test_training.py**
Unit tests for training scripts using `pytest`.

### **tests/test_inference.py**
Unit tests for inference scripts using `pytest`.

Run tests using:
```sh
pytest tests/
```

---

## Dependencies

Dependencies are listed in `requirements.txt`. Install them using:
```sh
pip install -r requirements.txt
```

---

## Usage

### **Training the Model**
```sh
python notebooks/train_model_py.py
```

### **Running Inference**
```sh
python notebooks/run_model_inference_py.py
```

### **Deploying Jobs**
```sh
python cli_tool/cli.py deploy
```

---

## Limitations
- This project utilized the free-tier Databricks edition, which presents comparatively more development challenges than the Enterprise edition.
- Two issues were discovered related to the free-tier Databricks edition. The details are as follows:
  - https://github.com/SagarBansal7/Wine_Quality_Prediction_Model/issues/4
  - https://github.com/SagarBansal7/Wine_Quality_Prediction_Model/issues/5
- Extensive research was conducted using the official documentation, and a series of resolutions were attempted to address the issues. However, since the documentation is tailored and tested for premium account setups, none of the solutions worked with the free-tier subscription. (The attempted resolutions are documented in the above links.)
- On a positive note, the issues themselves are not complex and should be resolved within a few minutes with a premium subscription account and/or custom compute configurations.

## Future Enhancements
- With additional time, setting up custom compute configuration and CLI usage can be explored.
- In addition to MLFlow, more comprehensive post-deployment monitoring tool and data quality check tools can be added.