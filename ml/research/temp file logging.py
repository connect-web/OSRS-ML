import tempfile
import pickle
import mlflow
"""
It's likely a good idea to log all the cross-validation metric results to Mlflow because

You might notice that certain models with specific params have something in common where they find highest accuracy across multiple different datasets
This will be an actual trend worth noting for future models also.

"""

result = gp_minimize(objective, space, n_calls=50, random_state=42)

# Using tempfile to create a temporary file that is automatically deleted
with tempfile.NamedTemporaryFile() as tmp:
    # Pickle the object to the temporary file
    pickle.dump(result, tmp)
    tmp.flush()  # Ensure all data is written to disk
    tmp.seek(0)  # Reset file pointer

    # Log the temporary file as an artifact to MLflow
    mlflow.log_artifact(tmp.name)

# No need to manually delete the file; it's automatically removed when the block exits
