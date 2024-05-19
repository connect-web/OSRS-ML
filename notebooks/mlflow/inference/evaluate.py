import pandas as pd
from rs_data import get_dataframe, SkillType
from rs_data import (PCA, TSNE, UMAP)
from rs_data.database.rs_processing import Leaderboards
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from rs_data import get_players, export_usernames

mlflow.set_tracking_uri("http://localhost:5000")
client = MlflowClient()
def get_data(activity, limit=10_000, offset=1000):
    df, _ = get_dataframe(activity, limit=limit, offset=offset)
    X = df.drop(columns=['Banned', 'pid'])
    y = df['Banned']
    return X, y, df['pid']

def get_training_data(activity):
    X, y, _ = get_data(activity, limit=1000, offset=0)
    return X, y

def get_model(activity):
    # Specify the model name and the stage
    model_name = f"{activity} + extra features"
    version = 2  # Specify the version number

    # Construct the model URI
    model_uri = f"models:/{model_name}/{version}"

    # Load the model
    model = mlflow.sklearn.load_model(model_uri)

    # Fit training data
    X, y = get_training_data(activity)
    model.fit(X, y)
    return model


def find_and_save_results(activity):
    X, y, pids = get_data(activity)
    model = get_model(activity)
    y_pred = model.predict(X)

    y_pred = pd.DataFrame(y_pred, columns=['Ban Prediction'])
    df_pred = pd.concat([y, y_pred, pids], axis=1)

    bot_users = df_pred[(df_pred['Banned'] == False) & (df_pred['Ban Prediction'] == True)]
    banned_bot_users = df_pred[(df_pred['Banned'] == True) & (df_pred['Ban Prediction'] == True)]

    stats = {
        "Activity": activity,
        'Unbanned': len(bot_users),
        'Banned': len(banned_bot_users),
    }

    pids = bot_users['pid'].tolist()
    players_df = get_players(pids)

    export_usernames(activity=activity,
                     filename="Top 10K Users Q1 2024",
                     usernames=players_df['Name'].tolist(),
                     directory="../../../"
                     )

    return stats


