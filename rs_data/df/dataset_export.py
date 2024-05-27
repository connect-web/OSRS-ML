import os
import pandas as pd
import json


DATASET_EXPORT = os.path.join('data', 'datasets')

def create_folder(full_folder_path: str):
    """
    Creates folder if it doesn't exist.
    :param full_folder_path: The target folder path.
    """
    if not os.path.exists(full_folder_path):
        os.mkdir(full_folder_path)


def export_dataset(activity_name: str, df: pd.DataFrame, preprocessor_meta: dict, file_prefix: str = "", directory: str = None):
    """
    Exports a dataset to a parquet file and json with the meta data for preprocessing.

    :param activity_name: The Runescape Activity.
    :param df: The dataframe of the dataset.
    :param preprocessor_meta: The meta data for the scaling methods.
    :param file_prefix: Can be used to store a different type of dataset.
    :param directory: None means the script is in the Project root directory.
    """


    if directory is None:
        folder_output_path = os.path.join(DATASET_EXPORT, activity_name)
    else:
        folder_output_path = os.path.join(directory, DATASET_EXPORT, activity_name)

    create_folder(folder_output_path)

    if file_prefix: file_prefix += '_' # add an underscore delimiter if there is a prefix.


    df.to_parquet(os.path.join(folder_output_path, f'{file_prefix}dataset.parquet'), engine='pyarrow')
    with open(os.path.join(folder_output_path, 'dataset.json'), 'w') as x:
        x.write(json.dumps(preprocessor_meta))

    print(f'Your {activity_name} dataset has been exported!')