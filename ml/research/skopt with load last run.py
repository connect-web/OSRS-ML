"""
If we run 50 runs to find the optimal

- then run a smaller value like 10-20 runs to find an even closer optimal
- it will search within the predefined "best" feature space to look at even smaller values

"""

from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args

# Example space
space = [
    Real(0.01, 0.5, name='learning_rate'),
    Integer(10, 1000, name='n_estimators'),
    Categorical(['gini', 'entropy'], name='criterion')
]

# Assuming you have loaded x0 and y0 from a file or database
x0 = [0.1, 100, 'gini']  # Last session's best parameters
y0 = -0.95  # Last session's best objective value

@use_named_args(space)
def objective(learning_rate, n_estimators, criterion):
    model = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, learning_rate=learning_rate)
    return -np.mean(cross_val_score(model, X, y, cv=5, scoring='accuracy'))

# Use these in your gp_minimize call
result = gp_minimize(
    objective,
    space,
    x0=x0,  # Initial parameter guess
    y0=[y0],  # Initial objective function value guess
    n_calls=50,
    random_state=42
)
