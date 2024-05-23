from ml.api import list_all_experiments

from rs_data.database.rs_processing import Leaderboards

skills = Leaderboards.get_skill_names(True)
minigames = Leaderboards.get_minigame_names()

def get_skill_experiments() -> dict:
    return load_experiment_type(skills=True)

def get_minigame_experiments() -> dict:
    return load_experiment_type(minigames=True)

def get_all_experiments() -> (dict,dict):
    return load_experiment_type()



def load_experiment_type(extra_features=True, skills=False, minigames=False):
    skill_experiments = {} # name : id
    minigame_experiments = {} # name : id

    # Separate experiment into skill , minigames
    for exp_id, exp_name in list_all_experiments():

        if not extra_features and isExtraFeatures(exp_name):
            continue # do not store experiment
        if extra_features and not isExtraFeatures(exp_name):
            continue # do not store experiment

        # Store experiment in appropriate dict
        if isSkill(exp_name):
            skill_experiments[exp_name] = exp_id
        else:
            minigame_experiments[exp_name] = exp_id

    if (skills and minigames) or (not skills and not minigames):
        return skill_experiments, minigame_experiments
    elif skills:
        return skill_experiments
    elif minigames:
        return minigame_experiments
    else:
        raise Exception("Unexpected result from load_experiment_type()")

def isSkill(name):
    return get_activity(name) == 'Skill'

def isExtraFeatures(name):
    return 'extra features' in name

def get_activity(name):
    for activity_name in minigames:
        if activity_name in name:
            return 'Minigame'
    for activity_name in skills:
        if activity_name in name:
            return 'Skill'
    return 'Unknown'

