
from ..helpers import format_decision
#%%
from showdown.battle import Battle
from showdown.engine.objects import StateMutator
from showdown.engine.select_best_move import pick_safest
from showdown.engine.select_best_move import get_payoff_matrix

import config

import logging
logger = logging.getLogger(__name__)



from data import types
from data import conditions
import itertools




#for reinforcement learning
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam


from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.preprocessing import OneHotEncoder
#%%

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

    def find_best_move(self):
        battles = self.prepare_battles(join_moves_together=True) #battles = battle.prepare_battles(join_moves_together=True)
        safest_move = pick_safest_move_from_battles(battles)
        return format_decision(self, safest_move)






"""
b = battle
state = b.create_state()
mutator = StateMutator(state)
user_options, opponent_options = b.get_all_options()

"""


#%%


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
            hp[i] = state.self.active.hp/state.self.active.maxhp
        if name in opponent_active:
            hp[i] = state.opponent.active.hp/state.opponent.active.maxhp
        if name in self_reserve:
            hp[i] = state_self_reserve[name].hp/state_self_reserve[name].maxhp
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

#%%
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



def get_minimum_state_array(state, battle, mutator, pokedex, all_move_json, types, conditions):
    #get the very minimum state: just the active pokemon, hp
    #pkmn_ind, hp
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

#%%
def determine_state_minimum(array, pokedex):
    n_hp = 2
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
    return s



#%%


def q_learning():
    #https://adventuresinmachinelearning.com/reinforcement-learning-tutorial-python-keras/
    # now execute the q learning
    y = 0.95
    eps = 0.5
    decay_factor = 0.999
    r_avg_list = []
    for i in range(num_episodes):
        s = env.reset()
        eps *= decay_factor
        if i % 100 == 0:
            print("Episode {} of {}".format(i + 1, num_episodes))
        done = False
        r_sum = 0
        while not done:
            if np.random.random() < eps:
                a = np.random.randint(0, 2)
            else:
                a = np.argmax(model.predict(np.identity(5)[s:s + 1]))
            new_s, r, done, _ = env.step(a)
            target = r + y * np.max(model.predict(np.identity(5)[new_s:new_s + 1]))
            target_vec = model.predict(np.identity(5)[s:s + 1])[0]
            target_vec[a] = target
            model.fit(np.identity(5)[s:s + 1], target_vec.reshape(-1, 2), epochs=1, verbose=0)
            s = new_s
            r_sum += r
        r_avg_list.append(r_sum / 1000)



"""
array = get_state_array(state, battle, mutator, pokedex, all_move_json, types, conditions) # array is state
array = get_minimum_state_array(state, battle, mutator, pokedex, all_move_json, types, conditions) # array is state

states = len(array) #It is the number of inputs of states


#number of actions should be the option to switch to any/all pokemon (about 1200 pokemon) and choose between all moves (about 800 moves)
#but then we have to constrain which moves and switches are available based on the team and moves
actions = len(pokedex) + len(all_move_json)
available_actions = 9 #number of actions can perform: select a move (4) switch to another pokemon (5)

num_inputs = states
num_actions = actions
num_hidden = 128

inputs = layers.Input(shape=(num_inputs,))
common = layers.Dense(num_hidden, activation="relu")(inputs)
action = layers.Dense(num_actions, activation="softmax")(common)
critic = layers.Dense(1)(common)

model = keras.Model(inputs=inputs, outputs=[action, critic])



optimizer = keras.optimizers.Adam(learning_rate=0.01)
huber_loss = keras.losses.Huber()
action_probs_history = []
critic_value_history = []
rewards_history = []
running_reward = 0
episode_count = 0

episode_reward = 0

















"""


"""
names_of_all_pokemon = get_names_of_all_pokemon(state)
hps = get_hp_of_all_pokemon(state, names_of_all_pokemon)
state_hps = tf.convert_to_tensor(hps)
state_hps = tf.expand_dims(state_hps, 0)

action_probs, critic_value = model(state_hps)

action = np.random.choice(num_actions, p= np.squeeze(action_probs) )

"""























"""


episode_reward = 0

def build_model(states, actions):
    #
    model = Sequential()
    model.add(Flatten(input_shape=(1,states)))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model


def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=500, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy,
                  nb_actions=actions, nb_steps_warmup=2, target_model_update=1e-2)
    return dqn



def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])


model = build_model(states, actions)
model.summary()
dqn = build_agent(model, actions)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])



"""




























