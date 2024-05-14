from copy import deepcopy
from Entities.helpers.Calculator import LevelCalculator
from Entities.helpers.Leaderboards import get_skill_names
calculator = LevelCalculator()

def calculate_overall(df, skills):
    zero_overall = df['Overall'] == 0
    df.loc[zero_overall, 'Overall'] = df.loc[zero_overall, skills].sum(axis=1)
    return df

def DF_Levels(df):
    df_levels = deepcopy(df)
    skills = get_skill_names()

    # Calculate Overall in case it is zero on hiscores.
    df_levels = calculate_overall(df_levels, skills)

    df_levels[skills] = df_levels[skills].map(calculator.get_level)
    return df_levels


def DF_Skill_Ratio(df):
    # Create a deep copy of the DataFrame to avoid modifying the original data
    df_ratios = deepcopy(df)

    # Assuming get_skill_names() returns a list of column names for skills
    skills = get_skill_names()

    # Modify 'Overall' where it's 0 to be the sum of skills
    df_ratios = calculate_overall(df_ratios, skills)


    for skill in skills:
        # Divide each skill level by the 'Overall' value to create a ratio
        df_ratios[skill] = df_ratios[skill] / df_ratios['Overall']

    df_ratios['Overall'] /= 200_000_000 # formatting the Overall XP as ratio of the maximum.

    return df_ratios
