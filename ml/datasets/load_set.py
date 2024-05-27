import os
import json
import pandas as pd

from .preprocessing import get_preprocessor

DATASET_EXPORT = os.path.join('data', 'datasets')

def load_set(activity_name: str, file_prefix: str = "", directory: str = None):
    if directory is None:
        folder_path = os.path.join(DATASET_EXPORT, activity_name)
    else:
        folder_path = os.path.join(directory, DATASET_EXPORT, activity_name)

    if file_prefix: file_prefix += '_'  # add an underscore delimiter if there is a prefix.

    file_path = os.path.join(folder_path, f'{file_prefix}dataset')

    df = pd.read_parquet(f'{file_path}.parquet', engine='pyarrow')

    with open(f'{file_path}.json','r') as x:
        preprocessor_meta = json.loads(x.read())

    preprocessor = get_preprocessor(preprocessor_meta)

    return df,preprocessor

