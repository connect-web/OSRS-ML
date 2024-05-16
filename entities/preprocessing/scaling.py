from sklearn.preprocessing import StandardScaler
import pandas as pd

def _stats_scaling(data: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """
    Scales the specified features of the data using standard scaling.
    """
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[features])
    return pd.DataFrame(scaled_data, columns=features)

def stats_scaling(df: pd.DataFrame, feature_groups: list[list[str]]):
    """
    Scales both regular and aggregated features separately and combines them.
    """
    if len(feature_groups) == 0:
        raise Exception("No features to scale.")
    elif len(feature_groups) == 1:
        return _stats_scaling(df, feature_groups[0])
    else:
        scaled_stats = [_stats_scaling(df, stat_group) for stat_group in feature_groups]
        return pd.concat(scaled_stats, axis=1)
