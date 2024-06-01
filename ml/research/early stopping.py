from skopt.callbacks import Callback
import numpy as np

class EarlyStopping(Callback):
    """Stop the optimization if there is no improvement within 'n_calls'."""
    def __init__(self, n_calls=10, tol=0.01):
        self.n_calls = n_calls
        self.tol = tol
        self.best_score = np.inf
        self.no_improve_count = 0

    def __call__(self, res):
        """Check if there has been an improvement."""
        # Check if the current score is better (lower, since minimizing) than the best found so far
        current_score = res.func_vals[-1]
        if current_score < self.best_score - self.tol:
            self.best_score = current_score
            self.no_improve_count = 0
        else:
            self.no_improve_count += 1

        # Stop if no improvement in the last 'n_calls'
        if self.no_improve_count >= self.n_calls:
            return True

    def reset(self):
        """Reset the no_improve_count and best_score."""
        self.no_improve_count = 0
        self.best_score = np.inf


if __name__ == "__main__":
    from skopt import gp_minimize

    # Define your search space and objective function here...

    # Initialize the early stopping callback
    early_stopping_callback = EarlyStopping(n_calls=10, tol=0.01)

    # Run optimization with the early stopping callback
    result = gp_minimize(
        objective,
        space,
        n_calls=50,
        random_state=42,
        callback=[early_stopping_callback]
    )
