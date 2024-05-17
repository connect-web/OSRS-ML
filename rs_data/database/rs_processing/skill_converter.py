from copy import deepcopy

from .leaderboards import Leaderboards, rename_player_live, rename_aggregate
from .level_calculator import calculator

def calculate_overall(df, skills, type=None):
    """
    Calculates the overall score for skills. Modifies the DataFrame in place.

    :param df: DataFrame containing skills data.
    :param skills: List of skill column names.
    :param type: Type of skills ('live' or 'aggregate'). If None, it assumes no suffix.
    :return: DataFrame with overall score calculated.
    """
    overall_column = 'Overall' if type is None else f'Overall_{type}'
    zero_overall = df[overall_column] == 0
    df.loc[zero_overall, overall_column] = df.loc[zero_overall, skills].sum(axis=1)
    return df

def df_levels(df, type=None):
    """
    Converts XP values to levels for skills in the DataFrame.

    :param df: DataFrame containing skills data.
    :param type: Type of skills ('live' or 'aggregate'). If None, it assumes no suffix.
    :return: DataFrame with skill levels.
    """
    df_levels = deepcopy(df)
    skills = Leaderboards.get_skill_names()
    skills_to_use = skills if type is None else rename_player_live(skills) if type == 'live' else rename_aggregate(skills)

    # Calculate Overall in case it is zero on hiscores.
    df_levels = calculate_overall(df_levels, skills_to_use, type)

    df_levels[skills_to_use] = df_levels[skills_to_use].map(calculator.get_level)
    return df_levels

def df_skill_ratio(df, type=None):
    """
    Calculates the ratio of each skill level to the overall level.

    :param df: DataFrame containing skills data.
    :param type: Type of skills ('live' or 'aggregate'). If None, it assumes no suffix.
    :return: DataFrame with skill ratios.
    """
    df_ratios = deepcopy(df)
    skills = Leaderboards.get_skill_names()
    if type is None:
        df_ratios = calculate_overall(df_ratios, skills)

        for skill in skills:
            df_ratios[skill] = df_ratios[skill] / df_ratios['Overall']

        df_ratios['Overall'] /= 200_000_000  # formatting the Overall XP as ratio of the maximum.
    else:
        live_skills = rename_player_live(skills)
        aggregate_skills = rename_aggregate(skills)

        # Calculate Overall in case it is zero on hiscores.
        df_ratios = calculate_overall(df_ratios, live_skills, 'live')
        df_ratios = calculate_overall(df_ratios, aggregate_skills, 'aggregate')

        for live_skill in live_skills:
            # Divide each skill level by the 'Overall' value to create a ratio
            df_ratios[live_skill] = df_ratios[live_skill] / df_ratios['Overall_live']

        for agg_skill in aggregate_skills:
            # Divide each skill level by the 'Overall' value to create a ratio
            df_ratios[agg_skill] = df_ratios[agg_skill] / df_ratios['Overall_aggregate']

        df_ratios['Overall_live'] /= 200_000_000  # formatting the Overall XP as ratio of the maximum.

    return df_ratios

def df_levels_basic(df):
    """
    Converts XP values to levels for skills in the DataFrame. Basic version without suffix handling.

    :param df: DataFrame containing skills data.
    :return: DataFrame with skill levels.
    """
    return df_levels(df, type=None)

def df_skill_ratio_basic(df):
    """
    Calculates the ratio of each skill level to the overall level. Basic version without suffix handling.

    :param df: DataFrame containing skills data.
    :return: DataFrame with skill ratios.
    """
    return df_skill_ratio(df, type=None)

def df_levels_advanced(df):
    """
    Converts XP values to levels for skills in the DataFrame. Advanced version with suffix handling.

    :param df: DataFrame containing skills data.
    :return: DataFrame with skill levels.
    """
    return df_levels(df, type='live')

def df_skill_ratio_advanced(df):
    """
    Calculates the ratio of each skill level to the overall level. Advanced version with suffix handling.

    :param df: DataFrame containing skills data.
    :return: DataFrame with skill ratios.
    """
    return df_skill_ratio(df, type='live')
