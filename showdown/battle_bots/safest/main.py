
from ..helpers import format_decision
#%%
from showdown.battle import Battle
from showdown.engine.objects import StateMutator
from showdown.engine.select_best_move import pick_safest
from showdown.engine.select_best_move import get_payoff_matrix

import config

import logging
logger = logging.getLogger(__name__)

from data import all_move_json
from data import pokedex

from data import types
from data import items
from data import abilities
from data import conditions
from data import reward_table
import itertools
import copy

import matplotlib.pyplot as plt
import seaborn as sns
from showdown.engine.helpers import normalize_name
#for reinforcement learning
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, InputLayer, Dropout, Input, Activation, MaxPooling2D, concatenate
from tensorflow.keras.optimizers import Adam

import pandas as pd

from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow.keras.backend as K
from sklearn.preprocessing import OneHotEncoder

abilities = [normalize_name(x) for x in abilities["name"] ]
#%

def prefix_opponent_move(score_lookup, prefix):
    new_score_lookup = dict()
    for k, v in score_lookup.items():
        bot_move, opponent_move = k
        new_opponent_move = "{}_{}".format(opponent_move, prefix)
        new_score_lookup[(bot_move, new_opponent_move)] = v

    return new_score_lookup


def pick_safest_move_from_battles(battles):
    all_scores = dict()
    for i, b in enumerate(battles): # i = 0; b = battles[i]
        state = b.create_state()
        mutator = StateMutator(state)
        user_options, opponent_options = b.get_all_options()
        logger.debug("Searching through the state: {}".format(mutator.state))
        scores = get_payoff_matrix(mutator, user_options, opponent_options, depth = config.search_depth, prune=True)

        prefixed_scores = prefix_opponent_move(scores, str(i))
        all_scores = {**all_scores, **prefixed_scores}

    decision, payoff = pick_safest(all_scores)
    bot_choice = decision[0]
    logger.debug("Reinforcement Learning Slection: {}, {}".format(bot_choice, payoff))
    return bot_choice


class BattleBot(Battle):
    def __init__(self, *args, **kwargs):
        super(BattleBot, self).__init__(*args, **kwargs)


    def find_best_move(self, model, state_table, action, reward_sum, s_memory,  episode, y, eps, decay_factor):
        #checkpoint_filepath ='models/checkpoint'
        #cp_callback = tf.keras.callbacks.ModelCheckpoint( filepath=checkpoint_filepath, save_weights_only=False, save_best_only=False)

        #battle = battle_copy
        state = self.create_state() #state = battle.create_state() #state = battle_copy.create_state()
        mutator = StateMutator(state) # mutator = StateMutator(state)
        user_options, opponent_options = self.get_all_options() # user_options, opponent_options = battle.get_all_options() # user_options, opponent_options = battle_copy.get_all_options()
        print(f"                Episode: {episode}; Epsilon: {np.round(eps,2)}")
        """
        print(model.optimizer.get_weights()[0])
        print(K.eval(model.optimizer.iterations))
        K.eval(model.optimizer.lr)
        K.eval(model.optimizer.iterations)

        """
        if self.turn == 1: #battle.turn == 1
            """
            y = 0.95
            eps = 0.5
            decay_factor = 0.9
            state = battle.create_state()
            mutator = StateMutator(state)
            user_options, opponent_options = battle.get_all_options()

            state_table = get_state_table_for_rewards(state, battle, mutator, pokedex, all_move_json, types, conditions)


            """
            s = get_state_array(state, self, mutator, pokedex, all_move_json, types, conditions, abilities, items)
            #s = get_state_array(state, battle, mutator, pokedex, all_move_json, types, conditions, abilities, items)
            s_memory[0] = s
            state_table = get_state_table_for_rewards(state, self, mutator, pokedex, all_move_json, types, conditions)
            #state_table = get_state_table_for_rewards(state, battle, mutator, pokedex, all_move_json, types, conditions)
        else:
            """
            previous_state_table = state_table
            state = battle.create_state()
            mutator = StateMutator(state)
            user_options, opponent_options = battle.get_all_options()
            s = get_state_array(state, battle, mutator, pokedex, all_move_json, types, conditions, abilities, items)

            current_state_table = get_state_table_for_rewards(state, battle, mutator, pokedex, all_move_json, types, conditions)
            reward = calculate_reward_from_state_table(previous_state_table, current_state_table, reward_table)
            """
            previous_state_table = state_table
            state = self.create_state()
            mutator = StateMutator(state)
            user_options, opponent_options = self.get_all_options()
            s = get_state_array(state, self, mutator, pokedex, all_move_json, types, conditions, abilities, items)
            current_state_table = get_state_table_for_rewards(state, self, mutator, pokedex, all_move_json, types, conditions)
            reward = calculate_reward_from_state_table(previous_state_table, current_state_table, reward_table)
            print(f"                Turn {self.turn-1} reward sum:    {np.round(reward,2)}:   {action}")
            state_table = current_state_table
            #update neural network based on reward
            #https://adventuresinmachinelearning.com/reinforcement-learning-tutorial-python-keras/


            #getting previous s and filling in next s
            if self.turn <= len(s_memory):
                s_memory[self.turn - 1] = s
                previous_s = s_memory[self.turn - 2]
            else:
                index = (self.turn-1) % 5
                s_memory[index] = s
                previous_s = s_memory[index - 1]

            action_previous = action

            action_index = determine_previous_action_indexes(action_previous, pokedex, all_move_json)


            s_5 = np.concatenate([s[4], s[5] ], axis = 0)
            target = reward + y * np.max(model.predict(  [ s[0].reshape(1,-1), s[1].reshape(1,-1), s[2].reshape(1,-1), s[3].reshape(1,-1), s_5.reshape(1,-1) ]  ))
            target_vec =model.predict(  [ s[0].reshape(1,-1), s[1].reshape(1,-1), s[2].reshape(1,-1), s[3].reshape(1,-1), s_5.reshape(1,-1) ]  )[0]
            target_vec[action_index] = target


            model.fit([ s[0].reshape(1,-1), s[1].reshape(1,-1), s[2].reshape(1,-1), s[3].reshape(1,-1), s_5.reshape(1,-1) ] , target_vec.reshape(-1, len(target_vec)), verbose=0)
            reward_sum += reward


        if np.random.random() < eps:
            a = np.random.randint(0, len(user_options))
            action = user_options[a]
        else:
            action = pick_action_RL(model, state, self, mutator, pokedex, all_move_json, types, conditions)
            #action = pick_action_RL(model, state, battle, mutator, pokedex, all_move_json, types, conditions)
        #battles = self.prepare_battles(join_moves_together=True) #battles = battle.prepare_battles(join_moves_together=True)
        #safest_move = pick_safest_move_from_battles(battles)
        best_move = format_decision(self, action) #best_move = format_decision(battle, action)
        return best_move, model, state_table, action, reward_sum, s_memory

    def initialize_battle(self, build_model_bool):
        state = self.create_state() # state = battle.create_state()  # state = battle_copy.create_state()
        mutator = StateMutator(state) # mutator = StateMutator(state)
        user_options, opponent_options = self.get_all_options() # user_options, opponent_options = battle.get_all_options() # user_options, opponent_options = battle_copy.get_all_options()

        s = get_state_array(state, self, mutator, pokedex, all_move_json, types, conditions, abilities, items, initialize = True)
        #s = get_state_array(state, battle_copy, mutator, pokedex, all_move_json, types, conditions, abilities, items, initialize = True)

        state_table = None
        action = None
        reward_sum = 0

        s_memory = [s,s,s,s,s] #remembers last 5 s's

        if build_model_bool:
            model = build_model(s, pokedex, all_move_json)
        else:
            model = None
        return model, state_table, action, reward_sum, s_memory

    def battle_win_or_lose_reward(self, winner, model, state_table, action, reward_sum, s_memory, episode, y ):
        """
        previous_state_table = state_table
        state = battle.create_state()
        mutator = StateMutator(state)
        user_options, opponent_options = battle.get_all_options()
        array = get_minimum_state_array(state, battle, mutator, pokedex, all_move_json, types, conditions)

        """

        if winner == self.opponent.account_name:
            person = "opponent"
            multiplier = 1
        else:
            person = "self"
            multiplier = -1
        previous_state_table = state_table
        state = self.create_state()
        mutator = StateMutator(state)
        user_options, opponent_options = self.get_all_options()

        s = get_state_array(state, self, mutator, pokedex, all_move_json, types, conditions, abilities, items)

        reward = reward_table["pokemon_all_faint"][person]* multiplier

        #getting previous s and filling in next s
        if self.turn <= len(s_memory):
            s_memory[self.turn - 1] = s
            previous_s = s_memory[self.turn - 2]
        else:
            index = (self.turn-1) % 5
            s_memory[index] = s
            previous_s = s_memory[index - 1]

        action_previous = action

        action_index = determine_previous_action_indexes(action_previous, pokedex, all_move_json)
        s_5 = np.concatenate([s[4], s[5] ], axis = 0)
        target = reward + y * np.max(model.predict(  [ s[0].reshape(1,-1), s[1].reshape(1,-1), s[2].reshape(1,-1), s[3].reshape(1,-1), s_5.reshape(1,-1) ]  ))
        target_vec =model.predict(  [ s[0].reshape(1,-1), s[1].reshape(1,-1), s[2].reshape(1,-1), s[3].reshape(1,-1), s_5.reshape(1,-1) ]  )[0]
        target_vec[action_index] = target
        """
        # adjust learning rate through internal calculations given by initial_epoch, which seems to help anecdodally
        set_learning_rate_to_default = "no"
        if episode >=0 and episode <1000:
            tau = int(episode/10) % (10)
            if tau == 0 or tau == 1 or tau == 2: set_learning_rate_to_default = "yes"
        elif episode >=1000:
            if episode %1000 == 0: a = 30
            if a >0: set_learning_rate_to_default = "yes"
            a = a - 1

        print(f"                Episode: {episode}; Learning max: {set_learning_rate_to_default}; Epsilon: {np.round(eps,2)}")
        if set_learning_rate_to_default == "yes":
            model.fit(s_previous.reshape(1,-1), target_vec.reshape(-1, len(target_vec)), epochs=1, verbose=0)
        else:
            model.fit(s_previous.reshape(1,-1), target_vec.reshape(-1, len(target_vec)), initial_epoch=episode, epochs=episode + 1, verbose=0)
        """

        model.fit([ s[0].reshape(1,-1), s[1].reshape(1,-1), s[2].reshape(1,-1), s[3].reshape(1,-1), s_5.reshape(1,-1) ] , target_vec.reshape(-1, len(target_vec)), verbose=0)
        reward_sum += reward
        turns = self.turn
        return model, reward_sum, turns
"""
b = battle
state = b.create_state()
mutator = StateMutator(state)
user_options, opponent_options = b.get_all_options()


format_decision(battle, safest_move)
"""


#%


#get the pokedex number and index
def get_pokemon_index_number(name, pokedex):
    """
    name = state.self.active.id
    from data import pokedex
    """
    if name in pokedex.keys(): #if the name exists, then find the pokedex number (the index in the json file, not the actual pokedex number)
        ind = list(pokedex.keys()).index(name) + 1
    else: ind = len(pokedex.keys()) +2 # if pokemon is not in pokedex, then make it the unknown index (the largest index + 1)
    return ind

def get_move_index_number(move, all_move_json):
    """
    from data import all_move_json
    move = state.self.active.moves[0]["id"]
    """
    if move in all_move_json.keys(): #if the name exists, then find the pokedex number (the index in the json file, not the actual pokedex number)
        ind = list(all_move_json.keys()).index(move) +1
    else: ind = len(all_move_json.keys()) +2# if pokemon is not in pokedex, then make it the unknown index (the largest index + 1)
    return ind

def get_move_set_index_number_and_pp(moves, all_move_json):
    """
    moves = new_state_dict["self"]["active"]["moves"]
    moves = new_state_dict["opponent"]["active"]["moves"]

    """
    if len(moves) > 0:
        keys = [moves[x]["id"] for x in range(len(moves))]
        indexes = [get_move_index_number(key, all_move_json) for key in keys]
        disabled = [moves[x]["disabled"] for x in range(len(moves))]
        pp = [moves[x]["current_pp"] for x in range(len(moves))]
    else:
        indexes = [0,0,0,0]
        disabled = [False, False, False, False]
        pp = [20, 20, 20, 20] #dummy moves where their pp is 20

    #if pkmn only has 1-3 moves, fill in zeros
    if len(moves) == 1:
        indexes = indexes + [0,0,0]
        disabled = disabled + [False, False, False]
        pp = pp + [20,20,20]
    if len(moves) == 2:
        indexes = indexes + [0,0]
        disabled = disabled + [ False, False]
        pp = pp + [20,20]
    if len(moves) == 3:
        indexes = indexes + [0]
        disabled = disabled + [ False]
        pp = pp + [20]

    for dis in range(len(disabled)):
        if disabled[dis] == False: disabled[dis] = 0
        else: disabled[dis] = 1
    return indexes, disabled, pp


def get_ability_index_number(ability, abilities):
    """
    name = state.self.active.id
    from data import pokedex
    """
    if ability == None: #if we do not know the ability currently (has not revealed itself)
        ind = 0
    elif ability in abilities:
        ind = abilities.index(ability) + 1 #adding 1, because index of 0 is None
    else: ind = len(abilities) + 2 # if there's a new ability not in the csv ability file (in data folder)
    return ind

def get_item_index_number(item, items):
    """

    new_state_dict
    item = new_state_dict["self"]["active"]["item"]
    item = new_state_dict["opponent"]["active"]["item"]
    items
    """

    if item == None:
        ind = 0
    elif item == "unknown_item": #if we do not know the item currently (has not revealed itself)
        ind = 1
    elif item in items:
        ind = items.index(item) + 2 #adding 1, because index of 0 is None
    else: ind = len(items) + 3 # if there's a new item not in the item file (in data folder)
    return ind

def get_status_index_number(status,  conditions):
    if status == None:
        ind = 0
    elif status in conditions["status"]:
        ind = conditions["status"].index(status) +1
    else:
        ind = len(conditions["status"]) + 2 #accounting for future unknown status (hence, the + 2)
    return ind

def get_volatile_index_number(volatile,  conditions):
    """
    volatile  =  new_state_dict["self"]["active"]["volatileStatus"]
    volatile = ["leechseed", "smackdown", "taunt", "bind"]
    """
    if len(volatile) == 0:
        ind = [0]
    else:
        ind = []
        for v in range(len(volatile)):
            if volatile[v] in conditions["volatile_status"]:
                ind.extend([conditions["volatile_status"].index(volatile[v]) +1])
            else:
                ind.extend([ len(conditions["volatile_status"]) + 2])
    return ind

def get_type_index(pokemon_type, types):
    types_combo = list(itertools.combinations(types, 2))
    types_all = list(zip(types))
    types_all.extend(types_combo )
    current_type = pokemon_type
    if len(current_type) == 1:
        current_type= list(zip(current_type))[0]
    if len(current_type) > 1:
        current_type.sort()
        current_type= tuple(current_type)
    if current_type in types_all:
        ind = types_all.index(current_type)
    else:
        ind = len(types_all) + 1
    return ind

def get_weather_index(weather, conditions):
    if weather == None:
        ind = 0
    elif weather in conditions["weather"]:
        ind = conditions["weather"].index(weather) + 1
    else:
        ind = len(conditions["weather"])+ 2
    return ind

def get_field_index(field, conditions):
    if field == None:
        ind = 0
    elif field in conditions["field"]:
        ind = conditions["field"].index(field) + 1
    else:
        ind = len(conditions["field"])+ 2
    return ind

def get_side_conditions(mutator_state_person):
    toxic_count = mutator_state_person.side_conditions["toxic_count"]
    spikes = mutator_state_person.side_conditions["spikes"]
    protect = mutator_state_person.side_conditions["protect"]
    stealthrock = mutator_state_person.side_conditions["stealthrock"]
    tailwind = mutator_state_person.side_conditions["tailwind"]
    reflect = mutator_state_person.side_conditions["reflect"]
    lightscreen = mutator_state_person.side_conditions["lightscreen"]
    auroraveil = mutator_state_person.side_conditions["auroraveil"]
    stickyweb = mutator_state_person.side_conditions["stickyweb"]
    toxicspikes = mutator_state_person.side_conditions["toxicspikes"]
    wish = mutator_state_person.wish

    return [toxic_count, spikes, protect, stealthrock, tailwind, reflect, lightscreen, auroraveil, stickyweb, toxicspikes, wish[0], wish[1]]

def get_trapped(battle):
    trapped_self = battle.user.trapped
    trapped_opponent = battle.opponent.trapped
    if trapped_self:
        trapped_self = 1
    else:
        trapped_self = 0
    if trapped_opponent:
        trapped_opponent = 1
    else:
        trapped_opponent = 0
    return [trapped_self, trapped_opponent]

def get_status_conditions(state_person, pokedex, conditions):
    status_active = state_person.active.status
    if status_active == None:
        status_active_ind = 0
    elif status_active in conditions["status"]:
        status_active_ind = conditions["status"].index(status_active) +1
    else:
        status_active_ind = len(conditions["status"]) + 2 #accounting for future unknown status (hence, the + 2)
    reserve_keys = state_person.reserve.keys()
    reserve_status = np.zeros((len(reserve_keys), 2))
    for i, p in enumerate(reserve_keys):
        name = state_person.reserve[p].id
        status = state_person.reserve[p].status
        if status == None:
            status_index = 0
        elif status in conditions["status"]:
            status_index = conditions["status"].index(status) +1
        else:
            status_index = len(conditions["status"]) + 2 #accounting for future unknown status (hence, the + 2)

        pokemon_ind = get_pokemon_index_number(name, pokedex)
        reserve_status[i,0] = pokemon_ind
        reserve_status[i,1] = status_index


    return status_active_ind, reserve_status


def get_names_of_all_pokemon(state):
    self_active = [state.self.active.id]
    state_self_reserve = state.self.reserve
    self_reserve = [state_self_reserve[key].id for key in state_self_reserve.keys()]

    state_opponent_reserve = state.opponent.reserve
    opponent_active = [state.opponent.active.id]
    opponent_reserve = [state_opponent_reserve[key].id for key in state_opponent_reserve.keys()]

    self_active.extend(self_reserve)
    opponent_active.extend(opponent_reserve)

    return self_active, opponent_active



def get_hp_of_all_pokemon(state, names_of_all_pokemon):
    hp = np.zeros(shape = len(names_of_all_pokemon))

    self_active = [state.self.active.id]
    state_self_reserve = state.self.reserve
    self_reserve = [state_self_reserve[key].id for key in state_self_reserve.keys()]

    opponent_active = [state.opponent.active.id]
    state_opponent_reserve = state.opponent.reserve
    opponent_reserve = [state_opponent_reserve[key].id for key in state_opponent_reserve.keys()]

    for i in range(len(names_of_all_pokemon)):
        name = names_of_all_pokemon[i]
        #find out if pokemon is active or reserve

        if name in self_active:
            #edge case where pokemon has fainted, and thus max hp is set to zero for some reason
            if state.self.active.maxhp == 0:
                max_hp = 1
            else:
                max_hp = state.self.active.maxhp
            hp[i] = (state.self.active.hp)/(max_hp)
        if name in opponent_active:
            hp[i] = state.opponent.active.hp/state.opponent.active.maxhp
        if name in self_reserve:
            #edge case where pokemon has fainted, and thus max hp is set to zero for some reason
            if state_self_reserve[name].maxhp == 0:
                max_hp = 1
            else:
                max_hp = state_self_reserve[name].maxhp
            hp[i] = state_self_reserve[name].hp/max_hp
        if name in opponent_reserve:
            hp[i] = state_opponent_reserve[name].hp/state_opponent_reserve[name].maxhp
    return hp

def get_types_array(names_pokemon, active, reserve):
       array = np.zeros( shape = (len(names_pokemon), 2)  )
       all_reserve = [reserve[key].id for key in reserve.keys()]
       for i, n in enumerate(names_pokemon):
           ind = get_pokemon_index_number(n, pokedex)
           if n in active.id:
               array[0, 1] = get_type_index(active.types, types)
               array[0, 0] =ind
           if n in all_reserve:
               ind = get_pokemon_index_number(n, pokedex)
               array[i, 0] = ind
               array[i, 1] = get_type_index(reserve[n].types, types)
       return array


def get_boost(pkmn):
    #pkmn = state.self.active
    boost = np.zeros(5)
    boost[0] = pkmn.attack_boost
    boost[1] = pkmn.defense_boost
    boost[2] = pkmn.special_attack_boost
    boost[3] = pkmn.special_defense_boost
    boost[4] = pkmn.speed_boost
    return boost


def get_boost_array(names_pokemon, active, reserve):
    array = np.zeros( shape = (len(names_pokemon), 6)  )
    all_reserve = [reserve[key].id for key in reserve.keys()]
    for i, n in enumerate(names_pokemon):
        ind = get_pokemon_index_number(n, pokedex)
        if n in active.id:
            array[0,1:] = get_boost(active)
            array[0, 0] = ind
        if n in all_reserve:
            ind = get_pokemon_index_number(n, pokedex)
            array[i, 0] = ind
            array[i,1:] = get_boost(reserve[n])
    return array

def get_pkmn_status_df(pkmn_dict, pokedex, all_move_json, abilities, types, items, conditions):
    df= pd.DataFrame(columns =[ "name", "id", "hp", "ability", "type", "item", "status", "volatile", "attack_boost", "defense_boost", "special_attack_boost",
                                           "special_defense_boost" ,"speed_boost", "accuracy_boost", "evasion_boost",
                                           "move_1", "move_1_disabled", "move_1_pp", "move_2", "move_2_disabled" , "move_2_pp", "move_3", "move_3_disabled", "move_3_pp", "move_4", "move_4_disabled" ,"move_4_pp"])
    moves = pkmn_dict["moves"]
    move_set_indexes, move_set_disabled, move_set_pp = get_move_set_index_number_and_pp(moves, all_move_json)

    #when pkmn faints, max hp is set to zero for some reason, so setting to 1 so not to divide by zero
    if pkmn_dict["maxhp"] == 0:
        hp_max = 1
    else:
        hp_max = pkmn_dict["maxhp"]
    d = dict( name = pkmn_dict["id"],
             id = get_pokemon_index_number( pkmn_dict["id"], pokedex),
             hp = pkmn_dict["hp"] / hp_max,
             ability = get_ability_index_number(pkmn_dict["ability"], abilities),
             type = get_type_index(pkmn_dict["types"], types)  ,
             item = get_item_index_number(pkmn_dict["item"], items),
             status = get_status_index_number( pkmn_dict["status"],  conditions),
             volatile = get_volatile_index_number( pkmn_dict["volatileStatus"],  conditions),
             attack_boost = pkmn_dict["attack_boost"],
             defense_boost = pkmn_dict["defense_boost"],
             special_attack_boost = pkmn_dict["special_attack_boost"],
             special_defense_boost = pkmn_dict["special_defense_boost"],
             speed_boost = pkmn_dict["speed_boost"],
             accuracy_boost = pkmn_dict["accuracy_boost"],
             evasion_boost = pkmn_dict["evasion_boost"],
             move_1 = move_set_indexes[0],
             move_1_disabled = move_set_disabled[0]   ,
             move_1_pp = move_set_pp[0],
             move_2 = move_set_indexes[1],
             move_2_disabled = move_set_disabled[1],
             move_2_pp = move_set_pp[1],
             move_3 = move_set_indexes[2],
             move_3_disabled = move_set_disabled[2],
             move_3_pp = move_set_pp[2],
             move_4 = move_set_indexes[3],
             move_4_disabled = move_set_disabled[3],
             move_4_pp = move_set_pp[3]
             )
    df = df.append( d , ignore_index=True)
    return df


def one_hot(name, index, pokedex, all_move_json, types, conditions, abilities, items):
    """
    index = df_active["id"][0]
    index = df_active["move_1"][0]
    df_active["volatile"][0] = [0,1]
    index =  df_active["volatile"][0]
    """
    if name == "id":
        one_hot = np.zeros(len(pokedex)+2)
        one_hot[index] = 1
        return one_hot
    if name == "ability":
        one_hot = np.zeros(len(abilities)+2)
        one_hot[index] = 1
        return one_hot
    if name == "type":
        types_combo = list(itertools.combinations(types, 2))
        types_all = list(zip(types))
        types_all.extend(types_combo )
        one_hot = np.zeros(len(types_all))
        one_hot[index] = 1
        return one_hot
    if name == "item":
        one_hot = np.zeros(len(items)+ 2)
        one_hot[index] = 1
        return one_hot
    if name == "status":
        one_hot = np.zeros(len(conditions["status"])+ 2)
        one_hot[index] = 1
        return one_hot
    if name == "volatile":
        one_hot = np.zeros(len(conditions["volatile_status"])+ 2)
        for v in range(len(index)):
            one_hot[v] = 1
        return one_hot
    if name == "move":
        one_hot = np.zeros(len(all_move_json)+2)
        one_hot[index] = 1
        return one_hot
    if name == "disabled":
        one_hot = np.zeros(2)
        one_hot[index] = 1
        return one_hot
    if name == "field":
        one_hot = np.zeros(len(conditions["field"]))
        one_hot[index] = 1
        return one_hot
    if name == "weather":
        one_hot = np.zeros(len(conditions["weather"]))
        one_hot[index] = 1
        return one_hot



#%
def get_state_array(state, battle, mutator, pokedex, all_move_json, types, conditions, abilities, items, initialize = False):
    """
    battle = battle_copy

    """
    #pkmn_ind, hp, type, status, boost_attack, boost_defend, boost_sp, boost_sd, boost_speed, boost_acc, boost_evasion, move1_id, ppx4
    if initialize: #initializing for first move with an blank array data
        array_active_categories = np.zeros(9806)
        array_active_numeric = np.zeros(24)
        array_reserve_categories = np.zeros(48490)
        array_reserve_numeric = np.zeros(25)
        array_side_conditions = np.zeros(24)
        array_field = np.zeros(13)
    else:

        #get active ddf
        df_active = pd.DataFrame(columns =[ "name", "id", "hp", "ability", "type", "item", "status", "volatile", "attack_boost", "defense_boost", "special_attack_boost",
                                           "special_defense_boost" ,"speed_boost", "accuracy_boost", "evasion_boost",
                                           "move_1", "move_1_disabled", "move_1_pp", "move_2", "move_2_disabled" , "move_2_pp", "move_3", "move_3_disabled", "move_3_pp", "move_4", "move_4_disabled" ,"move_4_pp"])

        new_state_dict = copy.deepcopy(eval(str(state)))

        df_active = df_active.append( get_pkmn_status_df(new_state_dict["self"]["active"], pokedex, all_move_json, abilities, types, items, conditions) , ignore_index=True)
        df_active = df_active.append( get_pkmn_status_df(new_state_dict["opponent"]["active"], pokedex, all_move_json, abilities, types, items, conditions) , ignore_index=True)



        #get reserve df
        df_reserve_self = pd.DataFrame(columns =[ "name", "id", "hp", "ability", "type", "item", "status",
                                           "move_1", "move_1_pp", "move_2", "move_2_pp", "move_3","move_3_pp", "move_4", "move_4_pp"])
        df_reserve_opponent = pd.DataFrame(columns =[ "name", "id", "hp", "ability", "type", "item", "status",
                                           "move_1", "move_1_pp", "move_2", "move_2_pp", "move_3","move_3_pp", "move_4", "move_4_pp"])

        reserve_keys_self = list(new_state_dict["self"]["reserve"].keys())
        reserve_keys_opponent = list(new_state_dict["opponent"]["reserve"].keys())
        remove_columns = ["volatile", "attack_boost", "defense_boost", "special_attack_boost",
                                           "special_defense_boost" ,"speed_boost", "accuracy_boost", "evasion_boost",
                                           "move_1_disabled",  "move_2_disabled" ,"move_3_disabled", "move_4_disabled" ]


        for r in range(5):
            if r < len(reserve_keys_self):
                d = get_pkmn_status_df(new_state_dict["self"]["reserve"][reserve_keys_self[r]], pokedex, all_move_json, abilities, types, items, conditions)
                d = d.drop(remove_columns, axis = 1)
            else: d =  dict(name = "NA", id = 0, hp = 0, ability = 0, type = 0, item = 0, status = 0,
                                       move_1 = 0, move_1_pp = 0, move_2 = 0, move_2_pp = 0, move_3 = 0, move_3_pp = 0, move_4 = 0, move_4_pp = 0)
            df_reserve_self = df_reserve_self.append(d, ignore_index=True)

        for r in range(5):
            if r < len(reserve_keys_opponent):
                d = get_pkmn_status_df(new_state_dict["opponent"]["reserve"][reserve_keys_opponent[r]], pokedex, all_move_json, abilities, types, items, conditions)
                d = d.drop(remove_columns, axis = 1)
            else: d =  dict(name = "NA", id = 0, hp = 0, ability = 0, type = 0, item = 0, status = 0,
                                           move_1 = 0, move_1_pp = 0, move_2 = 0, move_2_pp = 0, move_3 = 0, move_3_pp = 0, move_4 = 0, move_4_pp = 0)
            df_reserve_opponent = df_reserve_opponent.append(d, ignore_index=True)


        #get side conditions
        array_side_conditions = np.concatenate([get_side_conditions(mutator.state.self) , get_side_conditions(mutator.state.opponent)]    )

        #convert to one-hot-encoding
        categories = ["id", "ability", "type", "item", "status", "volatile", "move_1", "move_2", "move_3", "move_4", "move_1_disabled", "move_2_disabled", "move_3_disabled", "move_4_disabled"]
        names = ["id", "ability", "type", "item", "status", "volatile", "move", "move", "move", "move", "disabled", "disabled", "disabled", "disabled"]

        for a in range(2):
            for l in range(len(categories)):
                if a == 0 and l == 0:
                    array_active_categories = one_hot( names[l], df_active[categories[l] ][a], pokedex, all_move_json, types, conditions, abilities, items)
                else:
                    array_active_categories = np.concatenate([array_active_categories, one_hot(names[l], df_active[categories[l] ][a], pokedex, all_move_json, types, conditions, abilities, items) ], axis = 0)

        #get numerical
        categories = ["hp", "attack_boost", "defense_boost", "special_attack_boost", "special_defense_boost", "speed_boost", "accuracy_boost", "evasion_boost", "move_1_pp", "move_2_pp", "move_3_pp", "move_4_pp"]
        for a in range(2):
            for l in range(len(categories)):
                if a ==0 and l == 0:
                    array_active_numeric = np.array([df_active[categories[l]][a]])
                else:
                    array_active_numeric = np.concatenate([array_active_numeric,
                                    np.array(   [  df_active[categories[l]][a] ]     )      ]    )


        df_reserve = [df_reserve_self, df_reserve_opponent]
        #reserve
        #convert to one-hot-encoding
        categories = ["id", "ability", "type", "item", "status", "move_1", "move_2", "move_3", "move_4"]
        names = ["id", "ability", "type", "item", "status", "move", "move", "move", "move"]

        for r in range(2):
            for a in range(5):
                for l in range(len(categories)):
                    if a == 0 and l == 0 and r == 0:
                        array_reserve_categories = one_hot( names[l], df_reserve[r][categories[l] ][a], pokedex, all_move_json, types, conditions, abilities, items)
                    else:
                        array_reserve_categories = np.concatenate([array_reserve_categories, one_hot(names[l], df_reserve[r][categories[l] ][a], pokedex, all_move_json, types, conditions, abilities, items) ], axis = 0)


        #get numerical
        categories = ["hp", "move_1_pp", "move_2_pp", "move_3_pp", "move_4_pp"]
        for r in range(2):
            for a in range(5):
                for l in range(len(categories)):
                    if a ==0 and l == 0:
                        array_reserve_numeric = np.array([df_reserve[r][categories[l]][a]])
                    else:
                        array_reserve_numeric = np.concatenate([array_active_numeric,
                                        np.array(   [  df_reserve[r][categories[l]][a] ]     )      ]    )


        if new_state_dict["trickroom"]:
            trickroom = np.array([1])
        else:
            trickroom = np.array([0])
        wish = np.concatenate( [ np.array(new_state_dict["self"]["wish"]), np.array(new_state_dict["opponent"]["wish"]) ]  )
        land = np.concatenate([
        one_hot( "weather", get_weather_index(new_state_dict["weather"], conditions), pokedex, all_move_json, types, conditions, abilities, items),
        one_hot( "field", get_weather_index(new_state_dict["field"], conditions), pokedex, all_move_json, types, conditions, abilities, items)
        ])
        array_field = np.concatenate( [ land, wish, trickroom] )

    s = [array_active_categories, array_active_numeric, array_reserve_categories, array_reserve_numeric, array_side_conditions, array_field]

    return s



def get_minimum_state_array(state, battle, mutator, pokedex, all_move_json, types, conditions, initialize = False):
    """
    battle = battle_copy

    """
    #get the very minimum state: just the active pokemon, hp
    #pkmn_ind, hp
    if initialize: #initializing for first move with an blank array data
        array_wide = np.zeros( shape = (2, 2)   )
        array = array_wide.flatten()
    else:
        array_wide = np.zeros( shape = (2, 2)   ) # 9 if include boosts
        names_pokemon_self, names_pokemon_opponent = get_names_of_all_pokemon(state)
        #get pokemon index number, column 0
        pkmn_ind = np.zeros(12)
        for i, p in enumerate(names_pokemon_self):
            pkmn_ind[i] = get_pokemon_index_number(p, pokedex)
        for i, p in enumerate(names_pokemon_opponent):
            pkmn_ind[i+6] = get_pokemon_index_number(p, pokedex)
        array_wide[0,0 ] = pkmn_ind[0]
        array_wide[1,0 ] = pkmn_ind[6]
        #get hp, column 1
        hp_current_pokemon = get_hp_of_all_pokemon(state, names_pokemon_self + names_pokemon_opponent)
        #edge cases when either self or opponent do not have 6 pokemon, so pad zeros to hp
        hp_self = hp_current_pokemon[0:len(names_pokemon_self)]
        hp_opponent = hp_current_pokemon[6:6+len(names_pokemon_opponent)]
        hp = np.zeros(12)
        hp[0:6] = np.pad(hp_self, (0,6-len(hp_self)))
        hp[6:12] =  np.pad(hp_opponent, (0,6-len(hp_opponent)))
        array_wide[0,1 ] = hp[0]
        array_wide[1,1 ] = hp[6]
        array = array_wide.flatten()
    return array




def get_minimum_state_array_from_state_table(state_table):
    array_wide = np.zeros( shape = (2, 2)   )
    array_wide[0,0:2] = state_table[0][0,0:2]
    array_wide[1,0:2] = state_table[0][6,0:2]
    array = array_wide.flatten()
    return array

#%
def get_state_table_for_rewards(state, battle, mutator, pokedex, all_move_json, types, conditions):
    #pkmn_ind, hp, type, status, boost_attack, boost_defend, boost_sp, boost_sd, boost_speed, boost_acc, boost_evasion, move1_id, ppx4
    array_wide = np.zeros( shape = (12, 9)   ) # 9 if include boosts

    names_pokemon_self, names_pokemon_opponent = get_names_of_all_pokemon(state)

    #get pokemon index number, column 0
    pkmn_ind = np.zeros(12)
    for i, p in enumerate(names_pokemon_self):
        pkmn_ind[i] = get_pokemon_index_number(p, pokedex)
    for i, p in enumerate(names_pokemon_opponent):
        pkmn_ind[i+6] = get_pokemon_index_number(p, pokedex)
    array_wide[:,0 ] = pkmn_ind

    #get hp, column 1
    hp_current_pokemon = get_hp_of_all_pokemon(state, names_pokemon_self + names_pokemon_opponent)
    #edge cases when either self or opponent do not have 6 pokemon, so pad zeros to hp
    hp_self = hp_current_pokemon[0:len(names_pokemon_self)]
    hp_opponent = hp_current_pokemon[6:6+len(names_pokemon_opponent)]
    hp = np.zeros(12)
    hp[0:6] = np.pad(hp_self, (0,6-len(hp_self)))
    hp[6:12] =  np.pad(hp_opponent, (0,6-len(hp_opponent)))
    array_wide[:,1 ] = hp

    #get types, column 2
    types_array = get_types_array(names_pokemon_self, state.self.active,  state.self.reserve)
    for i in range(len(types_array)):
        array_wide[np.where(array_wide[:,0] == types_array[i,0])[0][0],2] = types_array[i,1]

    types_array = get_types_array(names_pokemon_opponent, state.opponent.active,  state.opponent.reserve)
    for i in range(len(types_array)):
        array_wide[np.where(array_wide[:,0] == types_array[i,0])[0][0],2] = types_array[i,1]

    #get status, column 3
    status_self = get_status_conditions(state.self, pokedex, conditions)
    status_opponent = get_status_conditions(state.opponent, pokedex, conditions)
    array_wide[0,3] = status_self[0]
    array_wide[6,3] = status_opponent[0]
    for i in range(len(status_self[1])):
        array_wide[np.where(array_wide[:,0] == status_self[1][i,0])[0][0],3] = status_self[1][i,1]
    for i in range(len(status_opponent[1])):
        array_wide[np.where(array_wide[:,0] == status_opponent[1][i,0])[0][0],3] = status_opponent[1][i,1]

    #get boost, column 4-8
    boost_array = get_boost_array(names_pokemon_self, state.self.active,  state.self.reserve)
    for i in range(len(boost_array)):
        array_wide[np.where(array_wide[:,0] == boost_array[i,0])[0][0],4:9] = boost_array[i,1:]
    boost_array = get_boost_array(names_pokemon_opponent, state.opponent.active,  state.opponent.reserve)
    for i in range(len(boost_array)):
        array_wide[np.where(array_wide[:,0] == boost_array[i,0])[0][0],4:9] = boost_array[i,1:]

    #get trapped
    trapped = get_trapped(battle)
    side_conditions_self = get_side_conditions(mutator.state.self)
    side_conditions_opponent = get_side_conditions(mutator.state.opponent)


    return array_wide, side_conditions_self, side_conditions_opponent, trapped



def align_state_tables(previous_state_table, current_state_table):
    previous_pokemon = previous_state_table[0]
    current_pokemon = current_state_table[0]
    table_pkmn = np.dstack([previous_pokemon, np.zeros(shape = previous_pokemon.shape)])
    ids = previous_pokemon[:,0]
    #separating into two parts: self and opponenet (in case both you and opponent have same pokemon, and thus pkmn ids)

    #edge case where urshifu appears for the first time and the name changes to urshifurapidstrike or something
    #edge case where pokemon changes for, and thus pokedex ID (like dynanimax, or urshifu to urshifurapidstrike)
    startv = [0, 6]
    endv =  [6, 12]
    for x in range(len(startv)):
        start = startv[x]
        end = endv[x]
        for i in np.arange(0,6):
            #check if any same pokedex number (the actual pokedex number, not the ID or index)
            keys = []
            for k in range(len(  current_pokemon[start:end,0]  )):
                keys.extend(  [list(pokedex.keys())[int(current_pokemon[start:end,0][k])]  ] )

            previous_key = list(pokedex.keys())[int(ids[start:end][i] )   ]
            if "num" in pokedex[previous_key].keys():
                previous_key_num = pokedex[previous_key]["num"]
                current_num = []
                for k in range(len(keys)):
                    if "num" in pokedex[keys[k]].keys():
                        current_num.extend(  [ pokedex[keys[k]]["num"]  ] )
                    else:
                        current_num.extend( [0]  )
                current_index = np.where(previous_key_num == np.array(current_num))[0][0]
                current_index
                ids[start:end][i] = current_pokemon[start:end,0][current_index]
    table_pkmn[:,0,0] = ids

    startv = [0, 6]
    endv =  [6, 12]
    for x in range(len(startv)):
        start = startv[x]
        end = endv[x]
        for i in np.arange(0,6):
            ind = np.where(current_pokemon[start:end,0] ==  ids[start:end][i] )[0][0] + start
            table_pkmn[ind,:,1] = current_pokemon[ind,:]
    return table_pkmn

def pokemon_fainted(table_pkmn):
    hp0 = table_pkmn[:,1,0]
    hp1 = table_pkmn[:,1,1]
    any_zero = np.where(hp1 == 0)[0]
    if len(any_zero)> 0:
        faint_ind = any_zero[np.where(hp0[any_zero] > 0)[0]]
    else:
        faint_ind = []
    return faint_ind


def pokemon_hp_change(table_pkmn):
    hp0 = table_pkmn[:,1,0]
    hp1 = table_pkmn[:,1,1]
    hp_change = hp1 - hp0
    return hp_change

def pokemon_status_change(table_pkmn):
    status0 = table_pkmn[:,3,0]
    status1 = table_pkmn[:,3,1]
    status_change = status1 - status0
    return status_change

def pokemon_side_conditions_change(table_side):
    side0 = table_side[0,:]
    side1 = table_side[1,:]
    return side1 - side0

def pokemon_fainted_all(table_pkmn, user = "self"):
    hp0 = table_pkmn[:,1,0]
    hp1 = table_pkmn[:,1,1]
    if user == "self":
        hp1 = hp1[:6]
    else:
        hp1 = hp1[6:]
    if all(hp1 == 0) :
        all_fainted = True
    else:
        all_fainted = False
    return all_fainted



def calculate_reward_from_state_table(previous_state_table, current_state_table, reward_table): #MOST IMPORTANT THING TO CHANGE AND ADJUST
    """
    calculate_reward_from_state_table(previous_state_table, current_state_table, reward_table)

    b = battle
    state = b.create_state()
    mutator = StateMutator(state)
    user_options, opponent_options = b.get_all_options()

    #previous_state_table = get_state_table_for_rewards(state, battle, mutator, pokedex, all_move_json, types, conditions)
    #current_state_table = get_state_table_for_rewards(state, battle, mutator, pokedex, all_move_json, types, conditions)
    """
    table_pkmn = align_state_tables(previous_state_table, current_state_table)
    table_side_self = np.vstack([previous_state_table[1], current_state_table[1] ])
    table_side_opponent = np.vstack([previous_state_table[2], current_state_table[2] ])
    table_trapped = np.vstack([previous_state_table[3], current_state_table[3] ])


    pokemon_faint = pokemon_fainted(table_pkmn)
    all_fainted_self = pokemon_fainted_all(table_pkmn, user = "self")
    all_fainted_opponent = pokemon_fainted_all(table_pkmn, user = "opponent")
    pokemon_hp_change_calc = pokemon_hp_change(table_pkmn)
    pokemon_status_change_calc = pokemon_status_change(table_pkmn)
    side_conditions_change_self = pokemon_side_conditions_change(table_side_self)
    side_conditions_change_opponent = pokemon_side_conditions_change(table_side_opponent)
    self_changes = [pokemon_faint, all_fainted_self, pokemon_hp_change_calc, pokemon_status_change_calc, side_conditions_change_self]
    opponent_changes = [pokemon_faint, all_fainted_opponent, pokemon_hp_change_calc, pokemon_status_change_calc, side_conditions_change_opponent]
    #calculate rewards
    reward_self = 0
    reward_opponent = 0

    #rewards from self
    reward_self = update_reward(self_changes, reward_self, table_pkmn, person = "self")
    print(f"                Reward from self:      {np.round(reward_self,2)}")
    reward_opponent = update_reward(opponent_changes, reward_opponent, table_pkmn, person = "opponent")
    print(f"                Reward from opponent:  {np.round(reward_opponent,2)}")
    reward = reward_self + reward_opponent
    return reward



def update_reward(changes, reward_stat, table_pkmn, person = "self"):
    #changes = self_changes; reward_stat =0
    pokemon_faint, all_fainted, pokemon_hp_change, pokemon_status_change, side_conditions_change = changes
    if person == "self":
        start = 0
        end = 6
        multiplier = 1
    else:
        start = 6
        end = 12
        multiplier = -1

    if len(pokemon_faint) > 0:
        if any(pokemon_faint <end):
            if any(pokemon_faint >=start):
                reward_stat += reward_table["pokemon_faint"][person]* multiplier
    if all_fainted:
        reward_stat += reward_table["pokemon_all_faint"][person]* multiplier
        print("\n\n\n\n\nALL POKEMON FAINT \n\n\n\n\n")
    reward_stat += (np.sum(pokemon_hp_change[start:end] * reward_table["hp_change_percent"][person])* multiplier)
    if any(pokemon_status_change[start:end] > 0):
        #figure out the status change
        ind = np.where(pokemon_status_change[start:end] > 0)[0]
        for i in range(len(ind)):
            stat = int(abs(table_pkmn[start:end, 3,1][ind[i]])) - 1
            stat_name = list(reward_table["status"][person].keys())[stat]
            reward_stat += reward_table["status"][person][stat_name]* multiplier
    if any(pokemon_status_change[start:end] < 0): #if pokemon are cured (like from Heal Bell, then you get a huge reward)
        #figure out the status change
        ind = np.where(pokemon_status_change[start:end] < 0)[0]
        for i in range(len(ind)):
            stat = int(abs(table_pkmn[start:end, 3,1][ind[i]])) - 1
            stat_name = list(reward_table["status"][person].keys())[stat]
            reward_stat += reward_table["status"][person][stat_name] * -multiplier *10
    if any(side_conditions_change[start:end] != 0):
        #figure out the status change

        #if side conditions > 0, then they are NEW side consitions. Hence spikes are present and you get a negative reward
        ind = np.where(side_conditions_change[start:end] > 0)[0]
        for i in range(len(ind)):
            side_condition_name = list(reward_table["side_conditions"][person].keys())[ind[i]]
            reward_stat += reward_table["side_conditions"][person][side_condition_name]* multiplier


        #if side conditions < 0, then they are side consitions that were removed. Hence spikes are removed due to defog, you get a positive reward
        ind = np.where(side_conditions_change[start:end] < 0)[0]
        for i in range(len(ind)):
            side_condition_name = list(reward_table["side_conditions"][person].keys())[ind[i]]
            reward_stat += reward_table["side_conditions"][person][side_condition_name]* multiplier * -1
    return reward_stat
#%
def determine_state_minimum(array, pokedex):
    n_hp = 10
    hp_bins = list(np.linspace(0,1,n_hp+1))[1:]
    hp_bins[-1] = 0.999 #changing last bin so that there is a state where we know a pkmn definitely has 100%hp (>99.9%)
    hp_bins.extend([1.0])
    n_hp = len(hp_bins)
    n_pkmn = len(pokedex.keys())
    pkmn = list(pokedex.keys())
    array
    #pkmn = pkmn[0:10]
    np.concatenate([np.array(pkmn), np.array(hp_bins), np.array(pkmn), np.array(hp_bins)     ])

    self_id = np.zeros(shape = n_pkmn).astype(int)
    self_id[   int(array[0])   ] = 1
    self_hp = np.zeros(shape = n_hp).astype(int)
    self_hp[np.argmax(array[1]<=hp_bins)] = 1


    opponent_id = np.zeros(shape = n_pkmn).astype(int)
    opponent_id[   int(array[0])   ] = 1
    opponent_hp = np.zeros(shape = n_hp).astype(int)
    opponent_hp[np.argmax(array[1]<=hp_bins)] = 1

    s = np.concatenate([self_id, self_hp, opponent_id, opponent_hp    ])
    s= s.flatten()
    return s

def determine_state(array, pokedex):
    n_hp = 10
    hp_bins = list(np.linspace(0,1,n_hp+1))[1:]
    hp_bins[-1] = 0.999 #changing last bin so that there is a state where we know a pkmn definitely has 100%hp (>99.9%)
    hp_bins.extend([1.0])
    n_hp = len(hp_bins)
    n_pkmn = len(pokedex.keys())
    pkmn = list(pokedex.keys())
    array
    #pkmn = pkmn[0:10]
    np.concatenate([np.array(pkmn), np.array(hp_bins), np.array(pkmn), np.array(hp_bins)     ])

    self_id = np.zeros(shape = n_pkmn).astype(int)
    self_id[   int(array[0])   ] = 1
    self_hp = np.zeros(shape = n_hp).astype(int)
    self_hp[np.argmax(array[1]<=hp_bins)] = 1


    opponent_id = np.zeros(shape = n_pkmn).astype(int)
    opponent_id[   int(array[0])   ] = 1
    opponent_hp = np.zeros(shape = n_hp).astype(int)
    opponent_hp[np.argmax(array[1]<=hp_bins)] = 1

    s = np.concatenate([self_id, self_hp, opponent_id, opponent_hp    ])
    s= s.flatten()
    return s

def determine_next_action_indexes(battle, pokedex, all_move_json):
    user_options, opponent_options = battle.get_all_options()
    pkmn = list(pokedex.keys())
    moves = list(all_move_json.keys())
    possible_pokemon_to_switch = []

    pkmn_and_moves = pkmn+moves
    possible_state_index_to_swtich = []
    for i, n in enumerate(user_options):
        if "switch" in n and n != "voltswitch" and n != "allyswitch" and n != "switcheroo":
            p = n.split('switch ')[1]
            if p in pkmn_and_moves:
                possible_state_index_to_swtich.append(pkmn_and_moves.index(p))
        else:
            if n in pkmn_and_moves:
                possible_state_index_to_swtich.append(pkmn_and_moves.index(n))
    return possible_state_index_to_swtich

def determine_previous_action_indexes(action_previous, pokedex, all_move_json):
    pkmn = list(pokedex.keys())
    moves = list(all_move_json.keys())
    possible_pokemon_to_switch = []

    pkmn_and_moves = pkmn+moves
    possible_state_index_to_swtich = []

    if "switch" in action_previous and action_previous != "voltswitch" and action_previous != "allyswitch" and action_previous != "switcheroo":
        p = action_previous.split('switch ')[1]
        if p in pkmn_and_moves:
            possible_state_index_to_swtich.append(pkmn_and_moves.index(p))
    else:
        if action_previous in pkmn_and_moves:
            possible_state_index_to_swtich.append(pkmn_and_moves.index(action_previous))
    return possible_state_index_to_swtich



#%
def build_model(s , pokedex, all_move_json):
    s_active_categories, s_active_numeric, s_reserve_categories, s_reserve_numeric, s_side_conditions, s_field = s
    number_of_actions = len(pokedex) + len(all_move_json)

    optimizer = keras.optimizers.Adam( beta_1 = 0.99, learning_rate = 0.05)

    inputA = Input(shape = (  len(s_active_categories), )   )
    inputB = Input(shape = (  len(s_active_numeric), )   )
    inputC = Input(shape = (  len(s_reserve_categories), )   )
    inputD = Input(shape = (  len(s_reserve_numeric), )   )
    inputE = Input(shape = (  len(s_side_conditions) +  len(s_field), )   )


    A = Dense(32, activation='relu')(inputA)
    A = Dropout(0.2)(A)
    A = Dense(32, activation='relu')(A)
    A = Model(inputs=inputA, outputs=A)


    B = Dense(24, activation='relu')(inputB)
    B = Model(inputs=inputB, outputs=B)


    C = Dense(12, activation='relu')(inputC)
    C = Dropout(0.2)(C)
    C = Dense(12, activation='relu')(C)
    C = Model(inputs=inputC, outputs=C)


    D = Dense(25, activation='relu')(inputD)
    D = Model(inputs=inputD, outputs=D)


    E = Dense(37, activation='relu')(inputE)
    E = Model(inputs=inputE, outputs=E)

    combined = concatenate([A.output, B.output, C.output, D.output, E.output])

    Z = Dense(32, activation="relu")(combined)
    Z = Dense(number_of_actions, activation="sigmoid")(Z)

    model = Model(inputs=[A.input, B.input, C.input, D.input, E.input], outputs=Z)

    model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
    print(model.summary())
    return model





def pick_action_RL(model, state, battle, mutator, pokedex, all_move_json, types, conditions):
    s = get_state_array(state, battle, mutator, pokedex, all_move_json, types, conditions, abilities, items)

    s_5 = np.concatenate([s[4], s[5] ], axis = 0)
    Q = model.predict(  [ s[0].reshape(1,-1), s[1].reshape(1,-1), s[2].reshape(1,-1), s[3].reshape(1,-1), s_5.reshape(1,-1) ]  )
    actions_possible = determine_next_action_indexes(battle, pokedex, all_move_json)
    action_ind = np.argmax(Q[0,actions_possible])
    user_options, opponent_options = battle.get_all_options()
    action = user_options[action_ind]
    return action

























