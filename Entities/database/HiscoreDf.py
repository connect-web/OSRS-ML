import pandas as pd
from Entities.database.db import Connection
from Entities.database.RowFormat import RowFormat
from Entities.helpers.SkillMod import DF_Levels

def HiscoreDf(pids: list[int]) -> pd.DataFrame:
    """
    :param pids: List of player id's from the database.
    :return: pandas dataframe including Player Levels & Minigames.
    """
    db = Connection(localhost=True)
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
    df = DF_Levels(df)

    df['Overall'] = df['Overall'].apply(lambda x: f"{x / 1_000_000:.2f}M")
    return df




