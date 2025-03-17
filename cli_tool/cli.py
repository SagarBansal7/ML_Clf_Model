import click
import os
from databricks_jobs import DatabricksJobManager
#dbutils.import_notebook("cli_tool")

# Load Databricks credentials from environment variables
DATABRICKS_HOST = os.getenv("DATABRICKS_HOST")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")

@click.group(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
def cli():
    """Databricks CLI Tool for Deploying Jobs."""
    pass

@click.command(context_settings=dict(ignore_unknown_options=True))
def deploy():
    """Deploys the Databricks jobs."""
    if not DATABRICKS_HOST or not DATABRICKS_TOKEN:
        click.echo("❌ Missing Databricks credentials. Set DATABRICKS_HOST and DATABRICKS_TOKEN.")
        return

    manager = DatabricksJobManager(DATABRICKS_HOST, DATABRICKS_TOKEN)
    
    click.echo("🚀 Creating Databricks jobs...")
    manager.create_training_job()
    manager.create_inference_job()
    click.echo("✅ Deployment completed successfully!")

cli.add_command(deploy)

if __name__ == "__main__":
    cli()
