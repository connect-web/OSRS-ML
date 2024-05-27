import pandas as pd
from copy import deepcopy

from .api import list_all_experiments, fetch_experiment_data, get_run_metrics


current_parameters = ['params.Classifier', 'params.Sampling', 'params.PCA_n_components']

def get_best_roc_auc_experiments():
    """
    Get the metrics for each experiment that has been run.

    :return: Dataframe of each experiment & it's metrics.
    """
    best_scores = []
    invalid_experiments = 0
    metrics_missing = 0

    total = list_all_experiments()
    for experiment in total:
        # Get all the iterations of experiments with different parameters.
        experiment_data = fetch_experiment_data(experiment_id=experiment[0])

        if experiment_data is None:
            invalid_experiments += 1
        else:
            # Make a copy of the best run
            if 'metrics.ROC-AUC' in experiment_data:
                best_roc_auc_row = experiment_data.loc[
                    experiment_data['metrics.ROC-AUC'] == experiment_data['metrics.ROC-AUC'].max()].copy()

                best_roc_auc_row.loc[:,'experiment_id'] = experiment[0]
                best_roc_auc_row.loc[:,'experiment_name'] = experiment[1]


                activity_name = experiment[1].split('Experience Model')[0]
                activity_name = activity_name.replace("'", '', -1)

                best_roc_auc_row.loc[:,'activity'] = activity_name

                best_scores.append(best_roc_auc_row)
            else:
                metrics_missing += 1
    df = pd.concat(best_scores).reset_index(drop=True)

    df['params.PCA_n_components'] = df['params.PCA_n_components'].astype(int)

    print(f'You have {len(df)} valid experiments & {invalid_experiments} invalid experiments & {metrics_missing} metrics were missing.')
    return df

def get_model_metrics(models: list[dict]) -> pd.DataFrame:
    missing_metric = 0
    rows = []

    for model in models:
        metrics = get_run_metrics(model['run_id'])
        if metrics:
            data = deepcopy(model)
            data.update(metrics)
            rows.append(data)
        else:
            missing_metric += 1

    df = pd.DataFrame(rows)
    print(f'You have {len(df)} metrics & {missing_metric} were missing')
    return df