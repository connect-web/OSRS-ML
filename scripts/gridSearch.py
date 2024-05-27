from ml import (
    GridSearchActivity,
    get_best_roc_auc_experiments
)

import pandas as pd

# Adjust display settings
pd.set_option('display.max_rows', None)  # Display all rows
pd.set_option('display.max_columns', None)  # Display all columns


def main():
    experiments = get_best_roc_auc_experiments()
    print(experiments)


def run_dks_experiment():
    activity_name = 'Dagannoth Prime'

    experiments = get_best_roc_auc_experiments()
    df = experiments[experiments['activity'] == activity_name]

    grid_searcher = GridSearchActivity(
        activity=activity_name,
        classifier_name=df['params.Classifier'].iloc[0],
        n_components=df['params.PCA_n_components'].iloc[0]
    )
    grid_searcher.run()

if __name__ == "__main__":
    run_dks_experiment()