from dataclasses import dataclass


def SkillToArray(user_skills, ratio=False):
    skills = [
        user_skills.get(skill, 0)
        for skill in Leaderboards.Skills
    ]
    if ratio is False:

        return skills
    # Ratios

    # Dropping overall XP just looking at ratios.
    skills = skills[1:len(skills)]
    total = sum(skills)
    return [skill/total if skill != 0 else 0 for skill in skills]

def SkillToCombatArray(user_skills, ratio=False):
    skills = [
        user_skills.get(skill, 0)
        for skill in Combat.Skills
    ]
    if ratio is False:
        return skills
    # Ratios

    total = sum(skills)
    return [skill/total if skill != 0 else 0 for skill in skills]

def MinigameToArray(user_minigames, ratio=False):
    minigames = [
            user_minigames.get(minigame, 0)
            for minigame in Leaderboards.Minigames
        ]
    if ratio is False:
        return minigames
    # Ratios
    total = sum(minigames)
    return [minigame / total if minigame != 0 else 0 for minigame in minigames]


class Combat:
    Skills = {
        'Attack', 'Defence', 'Strength', 'Hitpoints', 'Ranged', 'Prayer', 'Magic',
    }

def get_skill_names(keep_overall=False):
    skills = list(Leaderboards.Skills.keys())

    if 'Overall' in skills and keep_overall ==False:
        skills.remove('Overall')
    return skills
def get_minigames_names():
    Minigames = list(Leaderboards.Minigames.keys())
    return Minigames


class Leaderboards:
    Skills = {
        'Overall': '0' , 'Attack': '1' , 'Defence': '2' , 'Strength': '3' , 'Hitpoints': '4' , 'Ranged': '5' ,
        'Prayer': '6' , 'Magic': '7' , 'Cooking': '8' , 'Woodcutting': '9' , 'Fletching': '10' , 'Fishing': '11' ,
        'Firemaking': '12' , 'Crafting': '13' , 'Smithing': '14' , 'Mining': '15' , 'Herblore': '16' , 'Agility': '17' ,
        'Thieving': '18' , 'Slayer': '19' , 'Farming': '20' , 'Runecraft': '21' , 'Hunter': '22' , 'Construction': '23'
        }

    Minigames = {
        'Bounty Hunter - Hunter': '2' , 'Bounty Hunter - Rogue': '3' , 'Bounty Hunter (Legacy) - Hunter': '4' ,
        'Bounty Hunter (Legacy) - Rogue': '5' , 'Clue Scrolls (all)': '6' , 'Clue Scrolls (beginner)': '7' ,
        'Clue Scrolls (easy)': '8' , 'Clue Scrolls (medium)': '9' , 'Clue Scrolls (hard)': '10' ,
        'Clue Scrolls (elite)': '11' , 'Clue Scrolls (master)': '12' , 'LMS - Rank': '13' , 'PvP Arena - Rank': '14' ,
        'Soul Wars Zeal': '15' , 'Rifts closed': '16' , 'Abyssal Sire': '17' , 'Alchemical Hydra': '18' ,
        'Artio': '19' , 'Barrows Chests': '20' , 'Bryophyta': '21' , 'Callisto': '22' , "Calvar'ion": '23' ,
        'Cerberus': '24' , 'Chambers of Xeric': '25' , 'Chambers of Xeric: Challenge Mode': '26' ,
        'Chaos Elemental': '27' , 'Chaos Fanatic': '28' , 'Commander Zilyana': '29' , 'Corporeal Beast': '30' ,
        'Crazy Archaeologist': '31' , 'Dagannoth Prime': '32' , 'Dagannoth Rex': '33' , 'Dagannoth Supreme': '34' ,
        'Deranged Archaeologist': '35' , 'Duke Sucellus': '36' , 'General Graardor': '37' , 'Giant Mole': '38' ,
        'Grotesque Guardians': '39' , 'Hespori': '40' , 'Kalphite Queen': '41' , 'King Black Dragon': '42' ,
        'Kraken': '43' , "Kree'Arra": '44' , "K'ril Tsutsaroth": '45' , 'Mimic': '46' , 'Nex': '47' ,
        'Nightmare': '48' , "Phosani's Nightmare": '49' , 'Obor': '50' , 'Phantom Muspah': '51' , 'Sarachnis': '52' ,
        'Scorpia': '53' , 'Skotizo': '54' , 'Spindel': '55' , 'Tempoross': '56' , 'The Gauntlet': '57' ,
        'The Corrupted Gauntlet': '58' , 'The Leviathan': '59' , 'The Whisperer': '60' , 'Theatre of Blood': '61' ,
        'Theatre of Blood: Hard Mode': '62' , 'Thermonuclear Smoke Devil': '63' , 'Tombs of Amascut': '64' ,
        'Tombs of Amascut: Expert Mode': '65' , 'TzKal-Zuk': '66' , 'TzTok-Jad': '67' , 'Vardorvis': '68' ,
        'Venenatis': '69' , "Vet'ion": '70' , 'Vorkath': '71' , 'Wintertodt': '72' , 'Zalcano': '73' , 'Zulrah': '74',

        'Colosseum Glory': '123', # this is actually 17 now...
        'Deadman Points': '1',
        "League Points": '0',
        }



    # League points table is empty but actually does exist. Possible that it was just reset after the leagues or something...
    @staticmethod
    def getTableId (name , Skill = True):
        return Leaderboards.Skills.get(name) if Skill \
            else Leaderboards.Minigames.get(name)

@dataclass
class NameFilter:
    Skills : dict = None
    Minigames : dict = None

    def __post_init__(self):
        if self.Skills is None: self.Skills = Leaderboards.Skills
        if self.Minigames is None: self.Minigames = Leaderboards.Minigames