import os

import pandas as pd
import csv

USERNAME_EXPORT = os.path.join('data', 'username_export')

def create_folder(folder_name: str, directory: str = None):
    """
    Creates folder if it doesn't exist.
    :param folder_name: The activity name.
    :return: None
    """

    if directory is None:
        full_folder_path = os.path.join(USERNAME_EXPORT, folder_name)
    else:
        full_folder_path = os.path.join(directory, USERNAME_EXPORT, folder_name)


    if not os.path.exists(full_folder_path):
        os.mkdir(full_folder_path)

def export_usernames(filename: str, activity:str, usernames: list[str], directory: str = None):
    """
    Creates folder for activity if does not exist then overwrites a file in folder with a filename with the usernames in the given list.
    :param filename: The file name.
    :param activity: The activity relevant to the users.
    :param usernames: The list of usernames to be written to the file.
    :return: None
    """
    create_folder(activity, directory)

    print(f'Writing {len(usernames)} usernames to {filename}')

    if directory is None:
        file_path = os.path.join(USERNAME_EXPORT, activity, filename)
    else:
        file_path = os.path.join(directory, USERNAME_EXPORT, activity, filename)


    header = ['Username']

    # Writing to a CSV file
    with open(f'{file_path}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)  # Writing the header
        for username in usernames:
            writer.writerow([username])

def export_dataframe(df: pd.DataFrame, file_prefix: str, activity: str, directory: str = None):
    """
    Exports dataframe to file in the appropriate activity directory.
    :param df: A pandas dataframe containing relevant activity players with their username and banned status.
    :param file_prefix: The prefix for the file being written, for example high risk / medium risk / low risk.
    :param activity: The activity relevant to the players.
    :return: None
    """
    ban_types = [True, False]

    for ban_type in ban_types:
        usernames = df[df['Banned'] == ban_type]['Name'].tolist()
        ban_string = 'banned' if ban_type else 'unbanned'
        export_usernames(f'{file_prefix}_{ban_string}', activity, usernames, directory)