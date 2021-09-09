
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
from data import conditions
from data import reward_table
import itertools
import copy



#for reinforcement learning
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, InputLayer, Dropout
from tensorflow.keras.optimizers import Adam


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


    def find_best_move(self, model, state_table, action, reward_sum, episode = 1):
        #checkpoint_filepath ='models/checkpoint'
        #cp_callback = tf.keras.callbacks.ModelCheckpoint( filepath=checkpoint_filepath, save_weights_only=False, save_best_only=False)
        y = 0.95
        eps = 0.4
        decay_factor = 0.995
        state = self.create_state() #state = battle.create_state()
        mutator = StateMutator(state) # mutator = StateMutator(state)
        user_options, opponent_options = self.get_all_options() # user_options, opponent_options = battle.get_all_options()

        """
        print(model.optimizer.get_weights()[0])


        tf.keras.backend.set_value(model.optimizer.iterations, episode) #update optimizer iterations (which is the number of episodes, i.e. battles)
        print(model.optimizer.get_weights()[0])
        K.eval(model.optimizer.weights)

        print(model.summary())

        model.optimizers
        tf.keras.optimizers.Optimizer.get_weights()
        #print(f"user options: {user_options}")
        """
        if self.turn == 1: #battle.turn == 1
            """
            y = 0.95
            eps = 0.5
            decay_factor = 0.9
            state = battle.create_state()
            mutator = StateMutator(state)
            user_options, opponent_options = battle.get_all_options()
            array = get_minimum_state_array(state, battle, mutator, pokedex, all_move_json, types, conditions)
            state_table = get_state_table_for_rewards(state, battle, mutator, pokedex, all_move_json, types, conditions)
            s = determine_state_minimum(array, pokedex)

            """

            array = get_minimum_state_array(state, self, mutator, pokedex, all_move_json, types, conditions)
            state_table = get_state_table_for_rewards(state, self, mutator, pokedex, all_move_json, types, conditions)
            s = determine_state_minimum(array, pokedex)
        else:
            """
            previous_state_table = state_table
            state = battle.create_state()
            mutator = StateMutator(state)
            user_options, opponent_options = battle.get_all_options()
            array = get_minimum_state_array(state, battle, mutator, pokedex, all_move_json, types, conditions)
            current_state_table = get_state_table_for_rewards(state, battle, mutator, pokedex, all_move_json, types, conditions)
            reward = calculate_reward_from_state_table(previous_state_table, current_state_table, reward_table)
            """
            previous_state_table = state_table
            state = self.create_state()
            mutator = StateMutator(state)
            user_options, opponent_options = self.get_all_options()
            array = get_minimum_state_array(state, self, mutator, pokedex, all_move_json, types, conditions)
            current_state_table = get_state_table_for_rewards(state, self, mutator, pokedex, all_move_json, types, conditions)
            reward = calculate_reward_from_state_table(previous_state_table, current_state_table, reward_table)
            print(f"                Turn {self.turn-1} reward sum:   {np.round(reward,2)}:   {action}")
            state_table = current_state_table
            s = determine_state_minimum(array, pokedex)
            #update neural network based on reward
            #https://adventuresinmachinelearning.com/reinforcement-learning-tutorial-python-keras/
            array_previous = get_minimum_state_array_from_state_table(previous_state_table)
            s_previous = determine_state_minimum(array_previous, pokedex)
            action_previous = action

            action_index = determine_previous_action_indexes(action_previous, pokedex, all_move_json)
            target = reward + y * np.max(model.predict(s.reshape(1,-1)))
            target_vec = model.predict(s_previous.reshape(1,-1))[0]
            target_vec[action_index] = target
            model.fit(s_previous.reshape(1,-1), target_vec.reshape(-1, len(target_vec)), initial_epoch=episode, epochs=episode + 1, verbose=0)
            reward_sum += reward
        eps_new = eps * (decay_factor**episode)
        if np.random.random() < eps_new:
            a = np.random.randint(0, len(user_options))
            action = user_options[a]
        else:
            action = pick_action_RL(model, state, self, mutator, pokedex, all_move_json, types, conditions)
        #action = pick_action_RL(model, state, battle, mutator, pokedex, all_move_json, types, conditions)
        #battles = self.prepare_battles(join_moves_together=True) #battles = battle.prepare_battles(join_moves_together=True)
        #safest_move = pick_safest_move_from_battles(battles)
        best_move = format_decision(self, action) #best_move = format_decision(battle, action)
        return best_move, model, state_table, action, reward_sum

    def initialize_battle(self, build_model_bool):
        state = self.create_state() # state = battle.create_state()  # state = battle_copy.create_state()
        mutator = StateMutator(state) # mutator = StateMutator(state)
        user_options, opponent_options = self.get_all_options() # user_options, opponent_options = battle.get_all_options() # user_options, opponent_options = battle_copy.get_all_options()
        #print(f"user options: {user_options}")
        array = get_minimum_state_array(state, self, mutator, pokedex, all_move_json, types, conditions, initialize = True) #array = get_minimum_state_array(state, battle_copy, mutator, pokedex, all_move_json, types, conditions, initialize = True)
        state_table = None
        action = None
        reward_sum = 0
        s = determine_state_minimum(array, pokedex)
        if build_model_bool:
            model = build_model(s, pokedex, all_move_json)
        else:
            model = None
        return model, state_table, action, reward_sum

    def battle_win_or_lose_reward(self, winner, model, state_table, action, reward_sum ):
        """
        previous_state_table = state_table
        state = battle.create_state()
        mutator = StateMutator(state)
        user_options, opponent_options = battle.get_all_options()
        array = get_minimum_state_array(state, battle, mutator, pokedex, all_move_json, types, conditions)

        """
        y = 0.95
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
        array = get_minimum_state_array(state, self, mutator, pokedex, all_move_json, types, conditions)
        reward = reward_table["pokemon_all_faint"][person]* multiplier
        s = determine_state_minimum(array, pokedex)
        array_previous = get_minimum_state_array_from_state_table(previous_state_table)
        s_previous = determine_state_minimum(array_previous, pokedex)
        action_previous = action

        action_index = determine_previous_action_indexes(action_previous, pokedex, all_move_json)
        target = reward + y * np.max(model.predict(s.reshape(1,-1)))
        target_vec = model.predict(s_previous.reshape(1,-1))[0]
        target_vec[action_index] = target
        model.fit(s_previous.reshape(1,-1), target_vec.reshape(-1, len(target_vec)), epochs=1, verbose=0)
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
        ind = list(pokedex.keys()).index(name)
    else: ind = len(pokedex.keys()) # if pokemon is not in pokedex, then make it the unknown index (the largest index + 1)
    return ind

def get_move_index_number(move, all_move_json):
    """
    from data import all_move_json
    move = state.self.active.moves[0]["id"]
    """
    if move in all_move_json.keys(): #if the name exists, then find the pokedex number (the index in the json file, not the actual pokedex number)
        ind = list(all_move_json.keys()).index(move)
    else: ind = len(all_move_json.keys()) # if pokemon is not in pokedex, then make it the unknown index (the largest index + 1)
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

#%
def get_state_array(state, battle, mutator, pokedex, all_move_json, types, conditions):
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
    array_self_active = array_wide[0,:]
    array_opponent_active = array_wide[6,:]

    array_self = array_wide[1:6,0:4].flatten()
    array_opponent = array_wide[7:12,0:4].flatten()

    array_self = np.concatenate([array_self_active, array_self , get_side_conditions(mutator.state.self), np.array([trapped[0]]) ])
    array_opponent = np.concatenate([array_opponent_active, array_opponent , get_side_conditions(mutator.state.opponent),  np.array([trapped[1]]) ])

    array = np.concatenate([array_self, array_opponent])
    return array



def get_minimum_state_array(state, battle, mutator, pokedex, all_move_json, types, conditions, initialize = False):
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

    previous_state_table = get_state_table_for_rewards(state, battle, mutator, pokedex, all_move_json, types, conditions)
    current_state_table = get_state_table_for_rewards(state, battle, mutator, pokedex, all_move_json, types, conditions)
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
    print(f"                self reward:         {np.round(reward_self,2)}")
    reward_opponent = update_reward(opponent_changes, reward_opponent, table_pkmn, person = "opponent")
    print(f"                opponent reward:     {np.round(reward_opponent,2)}")
    reward = reward_self + reward_opponent
    return reward



def update_reward(changes, reward_stat, table_pkmn, person = "self"):

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
        ind = np.where(side_conditions_change[start:end] != 0)[0]
        for i in range(len(ind)):
            side_condition_name = list(reward_table["side_conditions"][person].keys())[ind[i]]
            reward_stat += reward_table["side_conditions"][person][side_condition_name]* multiplier
    return reward_stat
#%
def determine_state_minimum(array, pokedex):
    n_hp = 4
    hp_bins = list(np.linspace(0,1,n_hp+1))[1:]
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
def build_model(s, pokedex, all_move_json):
    number_of_actions = len(pokedex) + len(all_move_json)
    input_len = len(s)

    model = Sequential()
    model.add(InputLayer(batch_input_shape=(1, input_len)))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(number_of_actions, activation='sigmoid'))
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    print(model.summary())
    return model





def pick_action_RL(model, state, battle, mutator, pokedex, all_move_json, types, conditions):
    array = get_minimum_state_array(state, battle, mutator, pokedex, all_move_json, types, conditions) # array is state
    s = determine_state_minimum(array, pokedex)
    Q = model.predict(s.reshape(1,-1))
    actions_possible = determine_next_action_indexes(battle, pokedex, all_move_json)
    action_ind = np.argmax(Q[0,actions_possible])
    user_options, opponent_options = battle.get_all_options()
    action = user_options[action_ind]
    return action

























