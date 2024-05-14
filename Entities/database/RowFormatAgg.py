from Entities.helpers.Leaderboards import get_skill_names, get_minigames_names, SkillToArray, MinigameToArray


class RowFormatAdvanced:
    """
    Formats rows into a DataFrame suitable for models and further analysis.

    """
    def __init__(self, extra_features=None):
        self.extra_features = [] if extra_features is None else extra_features
        self.columns = self.get_columns()

    def player_live(self, names: list[str]) -> list[str]:
        """
        Adds the suffix _live to the names of the live player columns.
        :param names: Columns to re-name.
        :return: Renamed list of columns
        """
        return [f'{name}_live' for name in names]
    def aggregate(self, names: list[str]) -> list[str]:
        """
        Adds the suffix _aggregate to the names of the aggregate columns.
        :param names: Columns to re-name.
        :return: Renamed list of columns
        """
        return [f'{name}_aggregate' for name in names]


    def get_columns(self):
        """
        Gets all of the columns used in the DataFrame.

        :return: All columns related to the player.
        """
        self.skills = get_skill_names(keep_overall=True)
        self.minigames = get_minigames_names()

        self.live_skills = self.player_live(self.skills)
        self.live_minigames = self.player_live(self.minigames)

        self.agg_skills = self.aggregate(self.skills)
        self.agg_minigames = self.aggregate(self.minigames)

        live_stats = self.live_skills + self.live_minigames
        aggregated_stat_gains = self.agg_skills + self.agg_minigames

        columns = ['pid', 'Banned'] + self.extra_features + live_stats + aggregated_stat_gains
        return columns

    def format_row(self, row: tuple, skills_idx: int, minigames_idx: int, skills_idx2: int, minigames_idx2: int):
        """
        Format the row from a variable skills & minigames size into a fixed size array containing separate columns per skill, minigame
        for Player_Live stats and aggregated stat gains.

        :param row: The row to format.
        :param skills_idx: The index of the Player_Live Skills.
        :param minigames_idx: The index of the Player_Live Minigames.
        :param skills_idx2: The index of the Aggregated Skills.
        :param minigames_idx2: The index of the Aggregated Minigames.
        :return: Formatted row with a static defined column count.
        """
        skill_cols = SkillToArray(row[skills_idx])
        minigame_cols = MinigameToArray(row[minigames_idx])

        agg_skill_cols = SkillToArray(row[skills_idx2])
        agg_minigame_cols = MinigameToArray(row[minigames_idx2])

        avoid_indexes = [skills_idx, minigames_idx, skills_idx2, minigames_idx2]

        row = [element for i, element in enumerate(row) if i not in avoid_indexes]

        return tuple(row + skill_cols + minigame_cols + agg_skill_cols + agg_minigame_cols)


    def format_rows(self, rows: list, skills_idx: int, minigames_idx: int, skills_idx2: int, minigames_idx2: int) -> list:
        """
        Format an entire list of rows into a static column size.

        :param rows: The rows to format.
        :param skills_idx: The index of the Player_Live Skills.
        :param minigames_idx: The index of the Player_Live Minigames.
        :param skills_idx2: The index of the Aggregated Skills.
        :param minigames_idx2: The index of the Aggregated Minigames.
        :return: Formatted rows with a static defined column count.
        """
        rows_fixed = []
        for row in rows:
            row = self.format_row(row, skills_idx, minigames_idx, skills_idx2, minigames_idx2)
            rows_fixed.append(row)
        return rows_fixed
