import os
import json
import pandas as pd
from showdown.engine.helpers import normalize_name
#PWD = "/home/arevell/Documents/pokemon/showdown/data"
PWD = os.path.dirname(os.path.abspath(__file__))

move_json_location = os.path.join(PWD, 'moves.json')
with open(move_json_location) as f:
    all_move_json = json.load(f)

pkmn_json_location = os.path.join(PWD, 'pokedex.json')
with open(pkmn_json_location, 'r') as f:
    pokedex = json.loads(f.read())

random_battle_set_location = os.path.join(PWD, 'random_battle_sets.json')
with open(random_battle_set_location, 'r') as f:
    random_battle_sets = json.load(f)

types_json_location = os.path.join(PWD, 'types.json')
with open(types_json_location, 'r') as f:
    types = json.loads(f.read())
types = types["types"]


conditions_json_location = os.path.join(PWD, 'conditions.json')
with open(conditions_json_location, 'r') as f:
    conditions = json.loads(f.read())


reward_table_json_location = os.path.join(PWD, 'reward_table.json')
with open(reward_table_json_location, 'r') as f:
    reward_table = json.loads(f.read())

items_json_location = os.path.join(PWD, 'items.json')
with open(items_json_location, 'r') as f:
    items = json.loads(f.read())
items = items["items"]


abilities_csv_location = os.path.join(PWD, 'abilities.csv')
abilities = pd.read_csv(abilities_csv_location)
abilities = [normalize_name(x) for x in abilities["name"] ]



pokemon_sets = random_battle_sets
