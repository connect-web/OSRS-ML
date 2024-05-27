import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.compose import ColumnTransformer

from rs_data import get_dataframe, SkillType
from rs_data.database.rs_processing import Leaderboards
from rs_data.df.dataset_export import export_dataset


class ExportData:
    user_limit = 1000

    def __init__(self):
        self.activities =  Leaderboards.get_skill_names(keep_overall=True) + Leaderboards.get_minigame_names()
        print(f'You have {len(self.activities)} activities to complete!')

    def export_dataset(self, activity):
        df, formatter = get_dataframe(activity, limit=self.user_limit, aggregate=True)

        if len(df) < 1000:
            print(f'{activity} has only {len(df)} users...')
            return

        column_data = {
            'live': {'skills': formatter.live_skills, 'minigames': formatter.live_minigames},
            'aggregated': {'skills': formatter.agg_skills, 'minigames': formatter.agg_minigames},
            'time_features': formatter.extra_features
        }
        preprocessor_meta = {
            'columns': column_data,
            'processor_type': 'default'
        }
        # storing the pids within the dataset can help prevent using them for inference.
        export_dataset(activity, df, preprocessor_meta)


    def run(self):
        for activity in self.activities:
            self.export_dataset(activity)


if __name__ == '__main__':
    ExportData().run()
