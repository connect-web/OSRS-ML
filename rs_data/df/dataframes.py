from rs_data.database import (
    get_skill_hiscore
)

from .enums import SkillType


def get_dataframe(activity: str, limit: int, offset: int = 0, aggregate: bool = True,
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
    df, formatter = get_skill_hiscore(activity, limit=limit, offset=offset, aggregate=aggregate)

    # apply data processing
    df = skill_type.apply(df, aggregate)

    if extra_features is False:
        df = df.drop(columns=formatter.extra_features)

    return df, formatter
