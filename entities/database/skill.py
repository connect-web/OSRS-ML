import pandas as pd

from entities.database.connection import Connection
from entities.preprocessing.row_format import RowFormat

def get_skill_hiscore(activity, LIMIT = 500, OFFSET = 0):
    formatter = RowFormat()
    db = Connection(localhost=True)

    query = '''
    -- Fletching && Less than 5 Skills trained
    SELECT
        agg.pid,
        CASE WHEN nf.pid IS NULL THEN FALSE ELSE TRUE END as not_found,
        --AGG.skills ->> 'Fletching' as TargetActivity,
        pl.skills,
        pl.minigames
    
    from TASKS.aggregates agg
    LEFT JOIN player_live pl on pl.pid = agg.pid
    LEFT JOIN not_found nf on AGG.pid = nf.pid
    WHERE (AGG.skills ->> %s)::numeric IS NOT NULL
    -- and 5 > (SELECT count(*) FROM jsonb_object_keys(agg.skills) as keys)
    ORDER BY (AGG.skills ->> %s)::numeric DESC
    LIMIT %s
    OFFSET %s
    '''

    rows = db.get(query, params=tuple([activity]*2 + [LIMIT, OFFSET]))

    rows = formatter.format_rows(rows, skills_idx=2, minigames_idx=3)
    df = pd.DataFrame(rows, columns=formatter.get_columns())
    return df, formatter


def get_skill_hiscore_minimum_experience(activity, MINIMUM_EXPERIENCE=5_000_000, LIMIT=500, OFFSET=0):
    formatter = RowFormat()
    db = Connection(localhost=True)

    query = '''
    -- Fletching && Less than 5 Skills trained
    SELECT
        agg.pid,
        CASE WHEN nf.pid IS NULL THEN FALSE ELSE TRUE END as not_found,
        --AGG.skills ->> 'Fletching' as TargetActivity,
        pl.skills,
        pl.minigames

    from TASKS.aggregates agg
    LEFT JOIN player_live pl on pl.pid = agg.pid
    LEFT JOIN not_found nf on AGG.pid = nf.pid
    WHERE (AGG.skills ->> %s)::numeric IS NOT NULL
    AND (AGG.skills ->> %s)::numeric IS NOT NULL
    AND (AGG.skills ->> %s)::numeric > %s 

    -- and 5 > (SELECT count(*) FROM jsonb_object_keys(agg.skills) as keys)
    ORDER BY (AGG.skills ->> %s)::numeric DESC
    LIMIT %s
    OFFSET %s
    '''
    params = tuple(
        [activity] * 3 +
        [MINIMUM_EXPERIENCE] +
        [activity] +
        [LIMIT, OFFSET]
    )
    rows = db.get(query, params=params)

    rows = formatter.format_rows(rows, skills_idx=2, minigames_idx=3)
    df = pd.DataFrame(rows, columns=formatter.get_columns())
    return df, formatter