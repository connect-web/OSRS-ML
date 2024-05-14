from copy import deepcopy
from Entities.helpers.Calculator import LevelCalculator
from Entities.helpers.Leaderboards import get_skill_names
from Entities.database.RowFormatAgg import RowFormatAdvanced

calculator = LevelCalculator()

row_format = RowFormatAdvanced()

def calculate_overall(df, skills, type):
    zero_overall = df[f'Overall_{type}'] == 0
    df.loc[zero_overall, f'Overall_{type}'] = df.loc[zero_overall, skills].sum(axis=1)
    return df

def DF_Levels(df):
    df_levels = deepcopy(df)


    skills = get_skill_names()
    live_skills = row_format.player_live(skills)

    # Calculate Overall in case it is zero on hiscores.
    df_levels = calculate_overall(df_levels, live_skills, 'live')

    df_levels[live_skills] = df_levels[live_skills].map(calculator.get_level)
    return df_levels


def DF_Skill_Ratio(df):
    # Create a deep copy of the DataFrame to avoid modifying the original data
    df_ratios = deepcopy(df)

    # Assuming get_skill_names() returns a list of column names for skills
    skills = get_skill_names()
    live_skills = row_format.player_live(skills)
    aggregate_skills = row_format.player_live(skills)

    # Modify 'Overall' where it's 0 to be the sum of skills
    df_ratios = calculate_overall(df_ratios, live_skills, type='live')
    df_ratios = calculate_overall(df_ratios, aggregate_skills, type='aggregate')

    for live_skill in live_skills:
        # Divide each skill level by the 'Overall' value to create a ratio
        df_ratios[live_skill] = df_ratios[live_skill] / df_ratios['Overall_live']

    for live_skill in live_skills:
        # Divide each skill level by the 'Overall' value to create a ratio
        df_ratios[live_skill] = df_ratios[live_skill] / df_ratios['Overall_aggregate']


    df_ratios['Overall_live'] /= 200_000_000 # formatting the Overall XP as ratio of the maximum.
    return df_ratios
