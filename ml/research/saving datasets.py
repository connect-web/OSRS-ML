import pandas as pd
import os

"""
Have the preprocessor in a similar directory or have a class that stores:

- Directory
- Class method pertinent to the preprocessor


Datasets need to be saved in files so this can be run on the server
CPU is SPARE on server... less heat.


"""

def get_data(self, activity):
    df, formatter = get_dataframe(activity, limit=self.user_limit, aggregate=True, skill_type=self.skill_type)

    # Creating the ColumnTransformer
    preprocessor = ColumnTransformer(transformers=[
        ('std', StandardScaler(), formatter.agg_skills + formatter.agg_minigames),
        ('robust', RobustScaler(), formatter.extra_features),
        ('minmax', MinMaxScaler(), formatter.live_skills),
        ('minmax_2', MinMaxScaler(), formatter.live_minigames),
    ])

    X = df.drop(columns=['Banned', 'pid'])
    y = df['Banned']

    return X, y, preprocessor


def save_data_to_parquet(self, activity):
    X, y, preprocessor = self.get_data(activity)
    activity_folder = f"./data/{activity}"
    os.makedirs(activity_folder, exist_ok=True)  # Ensure the directory exists

    # Save X and y as Parquet files
    X.to_parquet(os.path.join(activity_folder, "features.parquet"))
    y.to_parquet(os.path.join(activity_folder, "labels.parquet"))

    # Optionally, serialize and save the preprocessor
    # This requires using joblib or pickle, since Parquet is not suitable
    import joblib
    joblib.dump(preprocessor, os.path.join(activity_folder, "preprocessor.joblib"))

    print(f"Data for {activity} saved in {activity_folder}")
