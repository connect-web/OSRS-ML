class Leaderboards:
    """
    Contains dictionaries of the Skills, Minigames as referenced by the Old school Runescape Hiscores.
    """
    SKILLS = {
        'Overall': '0', 'Attack': '1', 'Defence': '2', 'Strength': '3', 'Hitpoints': '4', 'Ranged': '5',
        'Prayer': '6', 'Magic': '7', 'Cooking': '8', 'Woodcutting': '9', 'Fletching': '10', 'Fishing': '11',
        'Firemaking': '12', 'Crafting': '13', 'Smithing': '14', 'Mining': '15', 'Herblore': '16', 'Agility': '17',
        'Thieving': '18', 'Slayer': '19', 'Farming': '20', 'Runecraft': '21', 'Hunter': '22', 'Construction': '23'
    }

    MINIGAMES = {
        'Bounty Hunter - Hunter': '2', 'Bounty Hunter - Rogue': '3', 'Bounty Hunter (Legacy) - Hunter': '4',
        'Bounty Hunter (Legacy) - Rogue': '5', 'Clue Scrolls (all)': '6', 'Clue Scrolls (beginner)': '7',
        'Clue Scrolls (easy)': '8', 'Clue Scrolls (medium)': '9', 'Clue Scrolls (hard)': '10',
        'Clue Scrolls (elite)': '11', 'Clue Scrolls (master)': '12', 'LMS - Rank': '13', 'PvP Arena - Rank': '14',
        'Soul Wars Zeal': '15', 'Rifts closed': '16', 'Abyssal Sire': '17', 'Alchemical Hydra': '18',
        'Artio': '19', 'Barrows Chests': '20', 'Bryophyta': '21', 'Callisto': '22', "Calvar'ion": '23',
        'Cerberus': '24', 'Chambers of Xeric': '25', 'Chambers of Xeric: Challenge Mode': '26',
        'Chaos Elemental': '27', 'Chaos Fanatic': '28', 'Commander Zilyana': '29', 'Corporeal Beast': '30',
        'Crazy Archaeologist': '31', 'Dagannoth Prime': '32', 'Dagannoth Rex': '33', 'Dagannoth Supreme': '34',
        'Deranged Archaeologist': '35', 'Duke Sucellus': '36', 'General Graardor': '37', 'Giant Mole': '38',
        'Grotesque Guardians': '39', 'Hespori': '40', 'Kalphite Queen': '41', 'King Black Dragon': '42',
        'Kraken': '43', "Kree'Arra": '44', "K'ril Tsutsaroth": '45', 'Mimic': '46', 'Nex': '47',
        'Nightmare': '48', "Phosani's Nightmare": '49', 'Obor': '50', 'Phantom Muspah': '51', 'Sarachnis': '52',
        'Scorpia': '53', 'Skotizo': '54', 'Spindel': '55', 'Tempoross': '56', 'The Gauntlet': '57',
        'The Corrupted Gauntlet': '58', 'The Leviathan': '59', 'The Whisperer': '60', 'Theatre of Blood': '61',
        'Theatre of Blood: Hard Mode': '62', 'Thermonuclear Smoke Devil': '63', 'Tombs of Amascut': '64',
        'Tombs of Amascut: Expert Mode': '65', 'TzKal-Zuk': '66', 'TzTok-Jad': '67', 'Vardorvis': '68',
        'Venenatis': '69', "Vet'ion": '70', 'Vorkath': '71', 'Wintertodt': '72', 'Zalcano': '73', 'Zulrah': '74',
        'Colosseum Glory': '123', 'Deadman Points': '1', "League Points": '0'
    }

    @staticmethod
    def get_skill_names(keep_overall=False):
        """
        Get the list of skill names.
        :param keep_overall: (bool, optional) Whether to include the 'Overall' skill in the array.
        :return: (array) The total list of skills.
        """
        skills = list(Leaderboards.SKILLS.keys())
        if not keep_overall and 'Overall' in skills:
            skills.remove('Overall')
        return skills

    @staticmethod
    def get_minigame_names():
        """
        Get the list of minigame names.
        :return: (array) The total list of minigames.
        """
        return list(Leaderboards.MINIGAMES.keys())

    @staticmethod
    def get_table_id(name, is_skill=True):
        """
        Returns the table id used in requests to get users from a Hiscore of the input skill/minigame.
        :param name: The name of the skill/minigame.
        :param is_skill: (bool, optional) Whether the name is of a skill. Default is True.
        :return: The table id.
        """
        return Leaderboards.SKILLS.get(name) if is_skill else Leaderboards.MINIGAMES.get(name)


class Combat:
    SKILLS = {'Attack', 'Defence', 'Strength', 'Hitpoints', 'Ranged', 'Prayer', 'Magic'}


def skill_to_array(user_skills, ratio=False):
    """
    Converts a dictionary of skills into a correctly padded array where values are not in the dictionary.
    :param user_skills: (dict) A dictionary of user skills.
    :param ratio: (bool, optional) Whether to return the ratio array instead of experience.
    :return: (array) The skills array.
    """
    skills = [user_skills.get(skill, 0) for skill in Leaderboards.SKILLS]
    if ratio:
        skills = skills[1:]  # Dropping overall XP just looking at ratios.
        total = sum(skills)
        return [skill / total if skill != 0 else 0 for skill in skills]
    return skills


def skill_to_combat_array(user_skills, ratio=False):
    """
    Converts a dictionary of combat skills into a correctly padded array where values are not in the dictionary.
    :param user_skills: (dict) A dictionary of user skills.
    :param ratio: (bool, optional) Whether to return the ratio array instead of experience.
    :return: (array) The combat skills array.
    """
    skills = [user_skills.get(skill, 0) for skill in Combat.SKILLS]
    if ratio:
        total = sum(skills)
        return [skill / total if skill != 0 else 0 for skill in skills]
    return skills


def minigame_to_array(user_minigames, ratio=False):
    """
    Converts a dictionary of minigames into a correctly padded array where values are not in the dictionary.
    :param user_minigames: (dict) A dictionary of user minigames & bosses.
    :param ratio: (bool, optional) Whether to return the ratio array instead of experience.
    :return: (array) The minigames array.
    """
    minigames = [user_minigames.get(minigame, 0) for minigame in Leaderboards.MINIGAMES]
    if ratio:
        total = sum(minigames)
        return [minigame / total if minigame != 0 else 0 for minigame in minigames]
    return minigames
