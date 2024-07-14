from rs_data.df.enums import SkillType
import pandas as pd


from rs_data.database.connection import Connection
from rs_data.database.rs_processing import RowFormat, RowFormatAdvanced, EXTRA_FEATURES
from rs_data.database.rs_processing.skill_converter import df_levels


def get_dataframe(primary_activity: str, limit: int, offset: int = 0, aggregate: bool = True,
                  skill_type: SkillType = SkillType.EXPERIENCE, extra_features=True):
    """
    Fetches users from the Hiscore rankings for Q1 2024 in the order of activity experience DESC.
    Rows are then converted into a dataframe and the formatter object is returned with the column names
    separated into group categories for scaling.

    :param activity: The skill activity.
    :param limit: The maximum amount of users.
    :param offset: The offset from the start of the users.
    :param aggregate: Optional boolean to return the Experience gained for Q1 2024. False will only include user experience at end of Q1 2024.
    :param skill_type: The type of processing to apply on skills, either ['EXPERIENCE', 'LEVELS', 'RATIO']
    :return:
    """


    df, formatter = get_hiscore(primary_activity,
                                limit=limit, offset=offset,
                                aggregate=aggregate,
                                activity_type='minigames')


    # apply data processing
    df = skill_type.apply(df, aggregate)

    if extra_features is False:
        df = df.drop(columns=formatter.extra_features)

    return df, formatter


def get_hiscore(activity, limit=500, offset=0, aggregate=False, activity_type='skills'):
    db = Connection(localhost=True)
    if aggregate:
        query = f'''
        SELECT agg.pid, CASE WHEN nf.pid IS NULL THEN FALSE ELSE TRUE END as not_found,
            pl.skills, pl.minigames, agg.skills, agg.minigames,
            agg.updates, agg.activescrapes, agg.inactivescrapes,
            agg.shortestinactivity, agg.shortestactivity,
            agg.longestinactivity, agg.longestactivity
        
        FROM TASKS.aggregates agg
        
        LEFT JOIN player_live pl on pl.pid = agg.pid
        LEFT JOIN not_found nf on agg.pid = nf.pid
        
        WHERE agg.HASNEGATIVE IS FALSE 
            AND 0 < COALESCE((agg.minigames -> 'Zulrah')::text::numeric, 0)
            AND 0 < COALESCE((agg.minigames -> 'Vorkath')::text::numeric, 0)
            AND 0 < COALESCE((agg.minigames -> %s)::text::numeric, 0)
            
            AND (agg.{activity_type} ->> %s)::numeric IS NOT NULL
            
        ORDER BY (agg.{activity_type} ->> %s)::numeric DESC
        LIMIT %s OFFSET %s
        '''
        formatter = RowFormatAdvanced(extra_features=EXTRA_FEATURES)
    else:
        query = f'''
        SELECT agg.pid, CASE WHEN nf.pid IS NULL THEN FALSE ELSE TRUE END as not_found,
            pl.skills, pl.minigames
        FROM TASKS.aggregates agg
        LEFT JOIN player_live pl on pl.pid = agg.pid
        LEFT JOIN not_found nf on agg.pid = nf.pid
        
        AND 0 < COALESCE((agg.minigames -> 'Zulrah')::text::numeric, 0)
        AND 0 < COALESCE((agg.minigames -> 'Vorkath')::text::numeric, 0)
        AND 0 < COALESCE((agg.minigames -> %s)::text::numeric, 0)
        
        WHERE (agg.{activity_type} ->> %s)::numeric IS NOT NULL
        ORDER BY (agg.{activity_type} ->> %s)::numeric DESC
        LIMIT %s OFFSET %s
        '''
        formatter = RowFormat()


    rows = db.get(query, params=(activity, activity, activity, limit, offset))
    db.close()

    if aggregate:
        rows = formatter.format_rows(rows, skills_idx=2, minigames_idx=3, skills_idx2=4, minigames_idx2=5)
    else:
        rows = formatter.format_rows(rows, skills_idx=2, minigames_idx=3)

    df = pd.DataFrame(rows, columns=formatter.get_columns())
    return df, formatter