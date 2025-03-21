import requests
import json
import os

class DatabricksJobManager:
    """Class to create and manage Databricks jobs."""

    def __init__(self, databricks_host, databricks_token):
        self.databricks_host = databricks_host
        self.databricks_token = databricks_token
        self.headers = {
            "Authorization": f"Bearer {self.databricks_token}",
            "Content-Type": "application/json"
        }

    def create_job(self, job_name, notebook_path, schedule=None):
        """Creates a Databricks job."""
        job_config = {
            "name": job_name,
            "tasks": [
                {
                    "task_key": job_name.replace(" ", "_").lower(),
                    "notebook_task": {
                        "notebook_path": notebook_path
                    },
                    "new_cluster": {
                        "spark_version": "11.3.x-scala2.12",
                        "num_workers": 2,
                        "node_type_id": "Standard_DS3_v2"
                    }
                }
            ]
        }

        if schedule:
            job_config["schedule"] = {
                "quartz_cron_expression": schedule,
                "timezone_id": "UTC"
            }

        response = requests.post(
            f"{self.databricks_host}/api/2.1/jobs/create",
            headers=self.headers,
            json=job_config
        )

        if response.status_code == 200:
            print(f"Job '{job_name}' created successfully!")
        else:
            print(f"Failed to create job '{job_name}': {response.text}")

    def create_training_job(self):
        """Creates the training job (Runs every 30 days)."""
        self.create_job(
            job_name="Train Classification Model",
            notebook_path="/Workspace/Users/sagarbansal719@gmail.com/ML_Clf_Model/notebooks/train_model_py.py",
            schedule="0 0 1 */1 *"  # Runs every 30 days
        )

    def create_inference_job(self):
        """Creates the inference job (Runs daily)."""
        self.create_job(
            job_name="Run Inference",
            notebook_path="/Workspace/Users/sagarbansal719@gmail.com/ML_Clf_Model/notebooks/run_model_inference_py.py",
            schedule="0 0 * * *"  # Runs daily
        )
