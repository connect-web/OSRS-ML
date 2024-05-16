import pandas as pd

from entities.database.connection import Connection
from entities.preprocessing.row_format import RowFormat

from entities.preprocessing.skill_converter import df_levels

db = Connection(localhost=True)

def HiscoreDf(pids: list[int]) -> pd.DataFrame:
    """
    :param pids: List of player id's from the database.
    :return: pandas dataframe including Player Levels & Minigames.
    """
    query = '''
    SELECT 
        PLAYERS.NAME,
        PLAYERS.ID,
        PLAYER_LIVE.SKILLS,
        PLAYER_LIVE.MINIGAMES
        
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




