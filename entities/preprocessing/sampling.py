import pandas as pd


def undersample(df):
    """
    Undersamples the majority class in a dataframe based on the specified limit.

    Parameters:
        df (DataFrame): The input DataFrame.
        banned_type (str): The column name indicating the class (e.g., 'Banned').
        limit (int): The maximum number of samples for the majority class after undersampling.

    Returns:
        DataFrame: A DataFrame where the majority class has been undersampled.
    """
    # Split the DataFrame into majority and minority classes
    df_a = df[df['Banned'] == 0]
    df_b = df[df['Banned'] == 1]

    limit = min([len(df_a),len(df_b)])

    if len(df_a) < len(df_b):
        majority_df = df_b
        minority_df = df_a
    else:
        majority_df = df_a
        minority_df = df_b

    # Check if the majority class needs to be undersampled
    if len(majority_df) > limit:
        # Randomly sample from the majority dataframe without replacement
        majority_df = majority_df.sample(n=limit, random_state=42)



    # Concatenate the minority and the undersampled majority dataframes
    undersampled_df = pd.concat([majority_df, minority_df], ignore_index=True)

    print(f'Undersampling complete: {len(df_a)+len(df_b)} into {len(undersampled_df)} rows.')
    print(f'Class size: {limit}')
    return undersampled_df