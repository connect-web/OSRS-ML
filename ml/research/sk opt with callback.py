from skopt.callbacks import DeltaYStopper


def mlflow_logging_callback(res):
    """Logs to MLflow after each iteration of the optimization process."""
    x_iters = res.x_iters  # List of input parameters for each iteration
    func_vals = res.func_vals  # List of objective function values of each iteration

    # Log the parameters and corresponding results of the latest iteration
    with mlflow.start_run():
        mlflow.log_params({name: value for name, value in zip([dim.name for dim in res.space.dimensions], x_iters[-1])})
        mlflow.log_metric("score", -func_vals[-1])  # Assuming minimization, negate the value if necessary


@use_named_args(space)
def objective(**params):
    # Here, assume a function `model_train_evaluate` trains the model and evaluates it
    score = model_train_evaluate(**params)
    return -score  # Negative because gp_minimize minimizes the objective


from skopt import gp_minimize

# Initialize MLflow run
mlflow.start_run()

# Execute the optimization with the callback included
result = gp_minimize(
    objective,
    space,
    n_calls=50,
    random_state=42,
    callback=[mlflow_logging_callback]
)

# Log the best overall performance
mlflow.log_metric("best_score", -result.fun)
mlflow.end_run()
