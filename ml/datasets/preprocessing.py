from rs_data.database.rs_processing import (
    EXTRA_FEATURES,
    Leaderboards,
    rename_aggregate,
    rename_player_live
)

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.compose import ColumnTransformer

SKILLS = Leaderboards.get_skill_names()
MINIGAMES = Leaderboards.get_minigame_names()

aggregate_skill_columns = rename_aggregate(SKILLS)
aggregate_minigame_columns = rename_aggregate(MINIGAMES)

live_skill_columns = rename_player_live(SKILLS)
live_minigame_columns = rename_player_live(MINIGAMES)

preprocessor_default = ColumnTransformer(transformers=[
    ('std', StandardScaler(), aggregate_skill_columns + aggregate_minigame_columns),
    ('robust', RobustScaler(), EXTRA_FEATURES),
    ('minmax', MinMaxScaler(), live_skill_columns),
    ('minmax_2', MinMaxScaler(), live_minigame_columns),
])

def default_preprocessor(columns: dict):
    return ColumnTransformer(transformers=[
        ('std', StandardScaler(), columns['aggregated']['skills'] + columns['aggregated']['minigames']),
        ('robust', RobustScaler(), EXTRA_FEATURES),
        ('minmax', MinMaxScaler(), columns['live']['skills']),
        ('minmax_2', MinMaxScaler(), columns['live']['minigames']),
    ])



PROCESSORS = {
    'default': lambda x: default_preprocessor(x)
}


def get_preprocessor(preprocessing_metadata: dict):
    processor_method = PROCESSORS.get(preprocessing_metadata['processor_type'])

    if processor_method is None:
        raise Exception(f"Unknown preprocessing meta type: {preprocessing_metadata['processor_type']}")

    return processor_method(preprocessing_metadata['columns'])

