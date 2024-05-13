from Entities.helpers.Leaderboards import get_skill_names, get_minigames_names, SkillToArray, MinigameToArray


class RowFormat:
    def __init__(self):
        self.columns = self.get_columns()

    def get_skill_names(self):
        return self.skills

    def get_minigame_names(self):
        return self.minigames

    def get_columns(self):
        self.skills = get_skill_names(keep_overall=True)
        self.minigames = get_minigames_names()

        columns = ['pid', 'Banned'] + self.skills + self.minigames
        return columns

    def format_row(self, row: tuple, skills_idx: int, minigames_idx: int):
        skill_cols = SkillToArray(row[skills_idx])
        minigame_cols = MinigameToArray(row[minigames_idx])

        row = [element for i, element in enumerate(row) if i not in [skills_idx, minigames_idx]]
        return tuple(row + skill_cols + minigame_cols)


    def format_rows(self, rows: list, skills_idx: int, minigames_idx: int) -> list:
        rows_fixed = []
        for row in rows:
            #print(row)
            row = self.format_row(row, skills_idx, minigames_idx)
            rows_fixed.append(row)
        return rows_fixed #[self.format_row(row, skills_idx, minigames_idx) for row in rows]
