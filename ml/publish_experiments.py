from .dataframes import get_best_roc_auc_experiments
from .api import (setup_environment_model)

def publish_best_experiments(version_alias):
    df = get_best_roc_auc_experiments()
    for index, row in df.iterrows():
        setup_environment_model(row['run_id'], row['activity'], version_alias)

