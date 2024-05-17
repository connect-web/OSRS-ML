import pandas as pd

from .connection import Connection
from .rs_processing import RowFormat, RowFormatAdvanced
from .rs_processing.skill_converter import df_levels


def get_skill_hiscore(activity, limit=500, offset=0, aggregate=False):
    db = Connection(localhost=True)

    if aggregate:
        extra_features = [
            'updates', 'activescrapes', 'inactivescrapes',
            'shortestinactivity', 'shortestactivity',
            'longestinactivity', 'longestactivity'
        ]
        query = '''
        SELECT agg.pid, CASE WHEN nf.pid IS NULL THEN FALSE ELSE TRUE END as not_found,
            pl.skills, pl.minigames, agg.skills, agg.minigames,
            agg.updates, agg.activescrapes, agg.inactivescrapes,
            agg.shortestinactivity, agg.shortestactivity,
            agg.longestinactivity, agg.longestactivity
        FROM TASKS.aggregates agg
        LEFT JOIN player_live pl on pl.pid = agg.pid
        LEFT JOIN not_found nf on agg.pid = nf.pid
        WHERE agg.HASNEGATIVE IS FALSE AND (agg.skills ->> %s)::numeric IS NOT NULL
        ORDER BY (agg.skills ->> %s)::numeric DESC
        LIMIT %s OFFSET %s
        '''
        formatter = RowFormatAdvanced(extra_features=extra_features)
    else:
        query = '''
        SELECT agg.pid, CASE WHEN nf.pid IS NULL THEN FALSE ELSE TRUE END as not_found,
            pl.skills, pl.minigames
        FROM TASKS.aggregates agg
        LEFT JOIN player_live pl on pl.pid = agg.pid
        LEFT JOIN not_found nf on agg.pid = nf.pid
        WHERE (agg.skills ->> %s)::numeric IS NOT NULL
        ORDER BY (agg.skills ->> %s)::numeric DESC
        LIMIT %s OFFSET %s
        '''
        formatter = RowFormat()

    rows = db.get(query, params=(activity, activity, limit, offset))
    db.close()

    if aggregate:
        rows = formatter.format_rows(rows, skills_idx=2, minigames_idx=3, skills_idx2=4, minigames_idx2=5)
    else:
        rows = formatter.format_rows(rows, skills_idx=2, minigames_idx=3)

    df = pd.DataFrame(rows, columns=formatter.get_columns())
    return df, formatter


def get_players(pids: list[int]) -> pd.DataFrame:
    """
    :param pids: List of player id's from the database.
    :return: pandas dataframe including Player Levels & Minigames.
    """
    db = Connection(localhost=True)
    query = '''
    SELECT 
        PLAYERS.NAME,PLAYERS.ID,PLAYER_LIVE.SKILLS,PLAYER_LIVE.MINIGAMES
    FROM PLAYER_LIVE
    LEFT JOIN PLAYERS ON PLAYERS.ID = PLAYER_LIVE.PID
    WHERE PLAYER_LIVE.PID = any(%s)
    '''

    rows = db.get(query, params=(pids,))

    row_format = RowFormat()

    formatted_row = row_format.format_rows(rows, 2, 3)
    columns = ['Name', 'pid'] + row_format.get_skill_names() + row_format.get_minigame_names()

    df = pd.DataFrame(formatted_row, columns=columns)
    df = df_levels(df)

    df['Overall'] = df['Overall'].apply(lambda x: f"{x / 1_000_000:.2f}M")
    return df
