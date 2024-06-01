from ml import (
    GridSearchActivity,
    get_best_roc_auc_experiments
)
from rs_data.database.rs_processing import Leaderboards


import pandas as pd

# Adjust display settings
pd.set_option('display.max_rows', None)  # Display all rows
pd.set_option('display.max_columns', None)  # Display all columns


def main():
    experiments = get_best_roc_auc_experiments()
    total_activities = Leaderboards.get_skill_names(keep_overall=True) + Leaderboards.get_minigame_names()

    for activity_name in total_activities:
        try:
            run_grid_search(experiments, activity_name)
        except Exception as e:
            print(f'{activity_name} : {e}')

def run_grid_search(experiments, activity_name, new_ban_data = False):
    df = experiments[experiments['activity'] == activity_name]

    grid_searcher = GridSearchActivity(
        activity=activity_name,
        classifier_name=df['params.Classifier'].iloc[0],
        n_components=df['params.PCA_n_components'].iloc[0],
        new_ban_data=new_ban_data
    )
    grid_searcher.run()

def run_dks_experiment():
    activity_name = 'Dagannoth Prime'
    experiments = get_best_roc_auc_experiments()

    run_grid_search(experiments, activity_name)

def run_dks_experiment_new_dataset():
    activity_name = 'Dagannoth Prime'
    experiments = get_best_roc_auc_experiments()

    run_grid_search(experiments, activity_name)



if __name__ == "__main__":
    run_dks_experiment_new_dataset()