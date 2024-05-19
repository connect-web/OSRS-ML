from enum import Enum

from rs_data.database.rs_processing import (
    df_levels_basic, df_levels_advanced,
    df_skill_ratio_basic, df_skill_ratio_advanced
)


class SkillType(Enum):
    EXPERIENCE = ("Experience", lambda df, aggregate: df)  # do nothing
    LEVELS = ("Levels", lambda df, aggregate: df_levels_advanced(df) if aggregate else df_levels_basic(df))
    RATIO = ("Ratio", lambda df, aggregate: df_skill_ratio_advanced(df) if aggregate else df_skill_ratio_basic(df))

    def __init__(self, description, func):
        self.description = description
        self.func = func

    def apply(self, df, aggregate=False):
        return self.func(df, aggregate)

if __name__ == "__main__":
    # Example usage:
    print(SkillType.EXPERIENCE)
    print(SkillType.LEVELS)
    print(SkillType.RATIO)

    # Creating a dummy DataFrame for example
    import pandas as pd
    df = pd.DataFrame({
        'skill1': [100, 200, 300],
        'skill2': [400, 500, 600]
    })

    # Applying lambda functions with aggregate = False
    print(f"{SkillType.EXPERIENCE.description} applied: {SkillType.EXPERIENCE.apply(df, aggregate=False)}")
    print(f"{SkillType.LEVELS.description} applied: {SkillType.LEVELS.apply(df, aggregate=False)}")
    print(f"{SkillType.RATIO.description} applied: {SkillType.RATIO.apply(df, aggregate=False)}")

    # Applying lambda functions with aggregate = True
    print(f"{SkillType.EXPERIENCE.description} applied: {SkillType.EXPERIENCE.apply(df, aggregate=True)}")
    print(f"{SkillType.LEVELS.description} applied: {SkillType.LEVELS.apply(df, aggregate=True)}")
    print(f"{SkillType.RATIO.description} applied: {SkillType.RATIO.apply(df, aggregate=True)}")
