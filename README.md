# ML_Clf_Model - Wine Quality Prediction Documentation

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

### 1. **Build**
- Runs on every push to `dev` or `main` branches.
- Sets up Python and installs dependencies.

### 2. **Deploy-Dev**
- Runs only on the `dev` branch.
- Runs tests and executes the CLI tool to deploy jobs to Databricks (development environment).

### 3. **Deploy-Main**
- Runs only on the `main` branch.
- Runs tests and executes the CLI tool to deploy jobs to Databricks (production environment).

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
- This project used free tier Databricks edition that has compartively more development challenges than Enterprise edition.
- Due to the serverless compute and time constraints, the GitHub Action workflow has issues running the cli tool.

## Future Enhancements
- With additional time, other free tier options can be explored that allows compute configuration and CLI usage.
- We can also explore the config and access enablement when running through Github Actions vs UI. 
- Current setup has dev and main(prod) branches. Stage can be added as an additional branch/env.
- In addition to MLFlow, more comprehensive post-deployment monitoring tool and data quality check tool can be added.
- Automation can be added to update the prod databricks files when a push to main(prod) branch has been made in git.



