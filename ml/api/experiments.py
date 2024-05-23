import mlflow
from mlflow.tracking import MlflowClient
mlflow.set_tracking_uri("http://localhost:5000")

# Initialize MLflow client
client = MlflowClient()

def list_all_experiments():
    # List all experiments
    experiments = mlflow.search_experiments()

    # Create a list of tuples (experiment_id, experiment_name)
    experiment_list = [(exp.experiment_id, exp.name) for exp in experiments]

    return experiment_list
def fetch_experiment_data(experiment_id):
    # Fetch all runs from a specific experiment
    runs = mlflow.search_runs(experiment_id)

    # Check if DataFrame is empty
    if runs.empty:
        #print("No data found for experiment ID:", experiment_id)
        return None

    return runs

def get_run_metrics(run_id) -> dict:
    run = client.get_run(run_id)
    return run.data.metrics if run is not None else None

def rename_experiment(experiment_id, new_experiment_name):
    # Fetch the current experiment details
    experiment = client.get_experiment(experiment_id)

    if experiment is None:
        raise ValueError(f"Experiment with ID {experiment_id} does not exist.")

    # Update the experiment name
    client.rename_experiment(experiment_id, new_experiment_name)

    print(f"Experiment ID {experiment_id} has been renamed to {new_experiment_name}")
