from .leaderboards import (
    Leaderboards, skill_to_array, minigame_to_array, rename_aggregate, rename_player_live
)

EXTRA_FEATURES = [
            'updates', 'activescrapes', 'inactivescrapes',
            'shortestinactivity', 'shortestactivity',
            'longestinactivity', 'longestactivity'
        ]

class RowFormatBase:
    """
    Base class for formatting rows into a DataFrame suitable for models and further analysis.
    """
    def __init__(self):
        self.columns = self.get_columns()

    def get_skill_names(self):
        return self.skills

    def get_minigame_names(self):
        return self.minigames

    def get_columns(self):
        self.skills = Leaderboards.get_skill_names(keep_overall=True)
        self.minigames = Leaderboards.get_minigame_names()
        return ['pid', 'Banned'] + self.skills + self.minigames

    def format_row(self, row, skills_idx, minigames_idx):
        skill_cols = skill_to_array(row[skills_idx])
        minigame_cols = minigame_to_array(row[minigames_idx])
        row = [element for i, element in enumerate(row) if i not in [skills_idx, minigames_idx]]
        return tuple(row + skill_cols + minigame_cols)

    def format_rows(self, rows, skills_idx, minigames_idx):
        return [self.format_row(row, skills_idx, minigames_idx) for row in rows]

class RowFormat(RowFormatBase):
    """
    Formats rows into a DataFrame suitable for models and further analysis.
    """
    def __init__(self):
        super().__init__()

class RowFormatAdvanced(RowFormatBase):
    """
    Formats rows with additional features into a DataFrame suitable for models and further analysis.
    """
    def __init__(self, extra_features=None):
        self.extra_features = [] if extra_features is None else extra_features
        super().__init__()

    def player_live(self, names):
        return [f'{name}_live' for name in names]

    def aggregate(self, names):
        return [f'{name}_aggregate' for name in names]

    def get_columns(self):
        self.skills = Leaderboards.get_skill_names(keep_overall=True)
        self.minigames = Leaderboards.get_minigame_names()

        self.live_skills = rename_player_live(self.skills)
        self.live_minigames = rename_player_live(self.minigames)

        self.agg_skills = rename_aggregate(self.skills)
        self.agg_minigames = rename_aggregate(self.minigames)

        live_stats = self.live_skills + self.live_minigames
        aggregated_stat_gains = self.agg_skills + self.agg_minigames

        return ['pid', 'Banned'] + self.extra_features + live_stats + aggregated_stat_gains

    def format_row(self, row, skills_idx, minigames_idx, skills_idx2, minigames_idx2):
        skill_cols = skill_to_array(row[skills_idx])
        minigame_cols = minigame_to_array(row[minigames_idx])

        agg_skill_cols = skill_to_array(row[skills_idx2])
        agg_minigame_cols = minigame_to_array(row[minigames_idx2])

        avoid_indexes = [skills_idx, minigames_idx, skills_idx2, minigames_idx2]
        row = [element for i, element in enumerate(row) if i not in avoid_indexes]

        return tuple(row + skill_cols + minigame_cols + agg_skill_cols + agg_minigame_cols)

    def format_rows(self, rows, skills_idx, minigames_idx, skills_idx2, minigames_idx2):
        return [self.format_row(row, skills_idx, minigames_idx, skills_idx2, minigames_idx2) for row in rows]
