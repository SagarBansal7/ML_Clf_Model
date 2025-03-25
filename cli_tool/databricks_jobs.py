import requests
import json
import os

class DatabricksJobManager:
    """Class to create and manage Databricks jobs."""

    def __init__(self, databricks_host, databricks_token, catalog, schema):
        self.databricks_host = databricks_host
        self.databricks_token = databricks_token
        self.catalog = catalog
        self.schema = schema
        self.params = { 
            "catalog": catalog,
            "schema": schema

        }
        self.headers = {
            "Authorization": f"Bearer {self.databricks_token}",
            "Content-Type": "application/json"
        }

    def create_job(self, job_name, notebook_path, schedule=None):
        """Creates a Databricks job."""
        # job_config = {
        #     "name": job_name,
        #     "tasks": [
        #         {
        #             "task_key": job_name.replace(" ", "_").lower(),
        #             "notebook_task": {
        #                  "notebook_path": notebook_path,
        #                  "base_parameters": self.params
        #             }
        #         }
        #     ]
        # }

        job_config = {
            "name": job_name,
            "tasks": [
                {
                    "task_key": job_name.replace(" ", "_").lower(),
                    "spark_python_task": {
                         "python_file": notebook_path,
                         "parameters": [self.catalog, self.schema]
                    },
                    "environment_key": "db_job_key"
                }
            ],
        "environments":[
            {   
                "environment_key":"db_job_key",
                "spec": {
                    "client": "2"
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
            f"{self.databricks_host}api/2.1/jobs/create",
            headers=self.headers,
            json=job_config
        )

        if response.status_code == 200:
            print(f"Job '{job_name}' created successfully!, {response}")
        else:
            print(f"Failed to create job '{job_name}': {response.text}")

    def create_training_job(self):
        """Creates the training job (Runs every 30 days)."""
        self.create_job(
            job_name="wine_quality_model_training_job",
            notebook_path="/Workspace/Users/sagarbansal719@gmail.com/Wine_Quality_Prediction_Model/notebooks/train_model_py.py",
            schedule="0 0 0 1 * ? *"  # Runs every 30 days
        )

    def create_inference_job(self):
        """Creates the inference job (Runs daily)."""
        self.create_job(
            job_name="wine_quality_model_inference_job",
            notebook_path="/Workspace/Users/sagarbansal719@gmail.com/Wine_Quality_Prediction_Model/notebooks/run_model_inference_py.py",
            schedule="0 0 0 * * ? *"  # Runs daily
        )
