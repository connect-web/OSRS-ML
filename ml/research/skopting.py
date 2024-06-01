from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args


"""
Logging the params will likely have to work differently this is a template

example:

result.x_iters I believe contains a list of param and results from all the iterations.

- These iterations will be good to store separately per-run

"""


# Define your search space
space = [
    Integer(10, 100, name='n_estimators'),
    Real(1e-6, 1e-1, prior='log-uniform', name='learning_rate')
]

# Define your objective function
@use_named_args(space)
def objective(**params):
    # Assuming a model function is already defined to train and evaluate the model
    return -model_train_evaluate(**params)

# First optimization
result = gp_minimize(objective, space, n_calls=50, random_state=42)

# Log the results in MLflow
with mlflow.start_run():
    mlflow.log_params(result.x_iters)
    mlflow.log_metrics({"best_score": -result.fun})
    mlflow.log_artifact("result.pkl")  # Assuming you save skopt results to a file
