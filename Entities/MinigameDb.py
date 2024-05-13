from Entities.database.db import Connection
from Entities.helpers.Leaderboards import SkillToArray, MinigameToArray, get_skill_names, get_minigames_names
import pandas as pd

class MinigameDb(Connection):
    def __init__(self, localhost):
        super().__init__(localhost)
    def get_multiple_bosses(self, minigame, limit, offset):
        query = '''
        SELECT
            AGG.pid,
            CASE WHEN nf.pid IS NULL THEN 'False' ELSE 'True' END as not_found,
            AGG.minigames ->> %s as TargetActivity,
            pl.skills,
            pl.minigames
            --agg.skills,
            --agg.minigames
            
            
        FROM
            statistics.players plrs
        LEFT JOIN tasks.aggregates AGG on AGG.pid = plrs.pid
        LEFT JOIN not_found nf on AGG.pid = nf.pid
        LEFT JOIN player_live pl on AGG.pid = pl.pid
    
        WHERE (AGG.minigames ->> %s)::numeric IS NOT NULL
          and 5 < (SELECT count(*)
               FROM jsonb_object_keys(agg.minigames) as keys)
        ORDER BY
          (AGG.minigames ->> %s)::numeric DESC
        LIMIT %s
        OFFSET %s;
        
        '''

        return self.get(query, params=(minigame,minigame,minigame,limit, offset))

    def get_induvidual_boss(self,minigame, limit , offset):
        query = '''
        SELECT
            AGG.pid,
            CASE WHEN nf.pid IS NULL THEN 'False' ELSE 'True' END as not_found,
            AGG.minigames ->> %s as TargetActivity,
            pl.skills,
            pl.minigames
            --agg.skills,
            --agg.minigames
            
            
        FROM
            statistics.players plrs
        LEFT JOIN tasks.aggregates AGG on AGG.pid = plrs.pid
        LEFT JOIN not_found nf on AGG.pid = nf.pid
        LEFT JOIN player_live pl on AGG.pid = pl.pid
    
        WHERE (AGG.minigames ->> %s)::numeric IS NOT NULL
          and 1 = (SELECT count(*)
               FROM jsonb_object_keys(agg.minigames) as keys)
        ORDER BY
          (AGG.minigames ->> %s)::numeric DESC
        LIMIT %s
        OFFSET %s;
        
        '''

        return self.get(query, params=(minigame,minigame,minigame,limit, offset))

class DataframeManager(MinigameDb):
    def __init__(self, localhost=False):
        super().__init__(localhost)
        self.skill_names = get_skill_names()
        self.minigame_names = get_minigames_names()
        self.column_names = ['pid', 'Banned', 'KC']
        self.column_names_experience = ['pid', 'Banned', 'KC', 'Overall']

        # 'Skills', 'Minigames', 'XP_Earned', 'Minigames_Earned'
        self.column_names.extend(self.skill_names)
        self.column_names_experience.extend(self.skill_names)

        #self.column_names.extend(self.minigame_names)

    @staticmethod
    def fix_rows(rows, ratio=True, minigames=False):
        """
        Converts skill, minigames && agg skill, agg minigames columns
        from variable length dict into static length array.
        :param rows:
        :return:
        """
        def fix_row(row, minigames=False):
            fixed = [row[0], row[1], row[2]]
            fixed.extend(SkillToArray(row[3], ratio=ratio))
            if minigames:
                fixed.extend(MinigameToArray(row[4], ratio=ratio))
            return tuple(fixed)

        return [fix_row(row, minigames) for row in rows]

    @staticmethod
    def fix_banned_column_type(df):
        df['Banned'] = df['Banned'].map({'True': True, 'False': False})
        df['Banned'] = df['Banned'].astype(bool)
        df['Banned'] = df['Banned'].fillna(False).astype(bool)
        return df

    def get_multiple_bosses(self, minigame, limit=5000, offset=0, ratio_mode=True):
        rows = super().get_multiple_bosses(minigame, limit, offset)
        columns = self.column_names if ratio_mode else self.column_names_experience
        df = pd.DataFrame(DataframeManager.fix_rows(rows, ratio=ratio_mode), columns=columns)
        df = DataframeManager.fix_banned_column_type(df)
        return df

    def get_induvidual_boss(self, minigame, limit=5000, offset=0, ratio_mode=True):
        rows = super().get_induvidual_boss(minigame, limit, offset)
        columns = self.column_names if ratio_mode else self.column_names_experience
        df = pd.DataFrame(DataframeManager.fix_rows(rows, ratio=ratio_mode), columns=columns)
        df = DataframeManager.fix_banned_column_type(df)
        return df