import pandas as pd

from entities.database.connection import Connection
from entities.preprocessing.row_format import RowFormatAdvanced

def get_skill_aggregate_hiscore(activity, LIMIT = 500, OFFSET = 0):

    db = Connection(localhost=True)

    query = '''
    -- Fletching && Less than 5 Skills trained
    SELECT
        agg.pid,
        CASE WHEN nf.pid IS NULL THEN FALSE ELSE TRUE END as not_found,
        --AGG.skills ->> 'Fletching' as TargetActivity,
        pl.skills,
        pl.minigames,
        agg.skills,
        agg.minigames,
        agg.updates,
        agg.activescrapes,
        agg.inactivescrapes,
        agg.shortestinactivity,
        agg.shortestactivity,
        agg.longestinactivity,
        agg.longestactivity
        
    from TASKS.aggregates agg
        LEFT JOIN player_live pl on pl.pid = agg.pid
        LEFT JOIN not_found nf on AGG.pid = nf.pid
    WHERE 
        AGG.HASNEGATIVE IS FALSE AND 
        (AGG.skills ->> %s)::numeric IS NOT NULL
        ORDER BY (AGG.skills ->> %s)::numeric DESC
    LIMIT %s
    OFFSET %s
    '''

    rows = db.get(query, params=tuple([activity]*2 + [LIMIT, OFFSET]))

    extra_features = [
        'updates',
        'activescrapes',
        'inactivescrapes',
        'shortestinactivity',
        'shortestactivity',
        'longestinactivity',
        'longestactivity'
    ]
    formatter = RowFormatAdvanced(extra_features=extra_features)

    rows = formatter.format_rows(rows, skills_idx=2, minigames_idx=3, skills_idx2=4, minigames_idx2=5)
    df = pd.DataFrame(rows, columns=formatter.get_columns())
    return df, formatter

