import importlib
import json
import asyncio
import concurrent.futures
from copy import deepcopy
import logging
import time
import random
import os.path
import numpy as np
import data
import pickle
from data.helpers import get_standard_battle_sets
import constants
import config
from showdown.engine.evaluate import Scoring
from showdown.battle import Pokemon
from showdown.battle import LastUsedMove
from showdown.battle_modifier import async_update_battle
from showdown.websocket_client import PSWebsocketClient
from tensorflow.keras.models import save_model, load_model
from numba import cuda
import tensorflow as tf


from data_analysis import data_analysis

import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import random

logger = logging.getLogger(__name__)

gpus = tf.config.experimental.list_physical_devices('GPU') #allowing GPU growth of memory
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)



def battle_is_finished(battle_tag, msg):
    return msg.startswith(">{}".format(battle_tag)) and constants.WIN_STRING in msg and constants.CHAT_STRING not in msg


async def async_pick_move(battle, model, state_table, action, reward_sum, episode):
    battle_copy = deepcopy(battle)
    if battle_copy.request_json:
        battle_copy.user.from_json(battle_copy.request_json)

    best_move, model, new_state_table, new_action, reward_sum = battle_copy.find_best_move(model, state_table, action, reward_sum, episode)
    #loop = asyncio.get_event_loop()
    #with concurrent.futures.ThreadPoolExecutor() as pool:
    #    best_move, model = await loop.run_in_executor(
    #        pool, battle_copy.find_best_move      )
    choice = best_move[0]
    if constants.SWITCH_STRING in choice:
        battle.user.last_used_move = LastUsedMove(battle.user.active.name, "switch {}".format(choice.split()[-1]), battle.turn)
    else:
        battle.user.last_used_move = LastUsedMove(battle.user.active.name, choice.split()[2], battle.turn)
    return best_move, model, new_state_table, new_action, reward_sum


async def handle_team_preview(battle, ps_websocket_client, build_model_bool):
    battle_copy = deepcopy(battle)
    battle_copy.user.active = Pokemon.get_dummy()
    battle_copy.opponent.active = Pokemon.get_dummy()

    """
    Hard coded technique:
        if zoroark and shedinja are in the team, start with zoroark and put shedinja last
        That way, shedinja appears first (but is really zoroark's illusion')
    """
    pokemon_names=[p.name for p in battle_copy.user.reserve]
    if "zoroark" in pokemon_names and 'shedinja' in pokemon_names:
        # Find zoroark's index
        ind_zoroark = pokemon_names.index("zoroark")+1
        ind_shedinja = pokemon_names.index("shedinja")+1
        model, state_table, action, reward_sum = battle_copy.initialize_battle(build_model_bool)

        best_move = f'/switch {ind_zoroark}'
        choice_digit = int(best_move.split()[-1])
        size_of_team = len(battle_copy.user.reserve) + 1
        team_list_indexes = list(range(1, size_of_team))
        team_list_indexes.remove(ind_zoroark)
        team_list_indexes.remove(ind_shedinja)
        message = ["/team {}{}{}|{}".format(ind_zoroark, "".join(str(x) for x in team_list_indexes), ind_shedinja, battle.rqid)]
    else:
        model, state_table, action, reward_sum = battle_copy.initialize_battle(build_model_bool)
        first_pokemon_sent_out = random.randint(1, 6)
        best_move = f'/switch {first_pokemon_sent_out}'
        size_of_team = len(battle.user.reserve) + 1
        team_list_indexes = list(range(1, size_of_team))
        choice_digit = int(best_move.split()[-1])
        team_list_indexes.remove(choice_digit)
        message = ["/team {}{}|{}".format(choice_digit, "".join(str(x) for x in team_list_indexes), battle.rqid)]

    battle.user.active = battle.user.reserve.pop(choice_digit - 1)
    await ps_websocket_client.send_message(battle.battle_tag, message)
    return model, state_table, action, reward_sum

async def get_battle_tag_and_opponent(ps_websocket_client: PSWebsocketClient):
    while True:
        msg = await ps_websocket_client.receive_message()
        split_msg = msg.split('|')
        first_msg = split_msg[0]
        if 'battle' in first_msg:
            battle_tag = first_msg.replace('>', '').strip()
            user_name = split_msg[-1].replace('☆', '').strip()
            opponent_name = split_msg[4].replace(user_name, '').replace('vs.', '').strip()
            return battle_tag, opponent_name


async def initialize_battle_with_tag(ps_websocket_client: PSWebsocketClient, set_request_json=True):
    battle_module = importlib.import_module('showdown.battle_bots.{}.main'.format(config.battle_bot_module))

    battle_tag, opponent_name = await get_battle_tag_and_opponent(ps_websocket_client)
    while True:
        msg = await ps_websocket_client.receive_message()
        split_msg = msg.split('|')
        if split_msg[1].strip() == 'request' and split_msg[2].strip():
            user_json = json.loads(split_msg[2].strip('\''))
            user_id = user_json[constants.SIDE][constants.ID]
            opponent_id = constants.ID_LOOKUP[user_id]
            battle = battle_module.BattleBot(battle_tag)
            battle.opponent.name = opponent_id
            battle.opponent.account_name = opponent_name

            if set_request_json:
                battle.request_json = user_json

            return battle, opponent_id, user_json


async def read_messages_until_first_pokemon_is_seen(ps_websocket_client, battle, opponent_id, user_json):
    # keep reading messages until the opponent's first pokemon is seen
    # this is run when starting non team-preview battles
    while True:
        msg = await ps_websocket_client.receive_message()
        if constants.START_STRING in msg:
            split_msg = msg.split(constants.START_STRING)[-1].split('\n')
            for line in split_msg:
                if opponent_id in line and constants.SWITCH_STRING in line:
                    battle.start_non_team_preview_battle(user_json, line)

                elif battle.started:
                    await async_update_battle(battle, line)

            # first move needs to be picked here
            best_move = await async_pick_move(battle)
            await ps_websocket_client.send_message(battle.battle_tag, best_move)

            return


async def start_random_battle(ps_websocket_client: PSWebsocketClient, pokemon_battle_type):
    battle, opponent_id, user_json = await initialize_battle_with_tag(ps_websocket_client)
    battle.battle_type = constants.RANDOM_BATTLE
    battle.generation = pokemon_battle_type[:4]

    await read_messages_until_first_pokemon_is_seen(ps_websocket_client, battle, opponent_id, user_json)

    return battle


async def start_standard_battle(ps_websocket_client: PSWebsocketClient, pokemon_battle_type, build_model_bool):
    battle, opponent_id, user_json = await initialize_battle_with_tag(ps_websocket_client, set_request_json=False)
    battle.battle_type = constants.STANDARD_BATTLE
    battle.generation = pokemon_battle_type[:4]

    if battle.generation in constants.NO_TEAM_PREVIEW_GENS:
        await read_messages_until_first_pokemon_is_seen(ps_websocket_client, battle, opponent_id, user_json)
    else:
        msg = ''
        while constants.START_TEAM_PREVIEW not in msg:
            msg = await ps_websocket_client.receive_message()

        preview_string_lines = msg.split(constants.START_TEAM_PREVIEW)[-1].split('\n')

        opponent_pokemon = []
        for line in preview_string_lines:
            if not line:
                continue

            split_line = line.split('|')
            if split_line[1] == constants.TEAM_PREVIEW_POKE and split_line[2].strip() == opponent_id:
                opponent_pokemon.append(split_line[3])

        battle.initialize_team_preview(user_json, opponent_pokemon)

        smogon_usage_data = get_standard_battle_sets(
            pokemon_battle_type,
            pokemon_names=[p.name for p in battle.opponent.reserve]
        )
        data.pokemon_sets = smogon_usage_data

        model, state_table, action, reward_sum = await handle_team_preview(battle, ps_websocket_client, build_model_bool)

    return battle, model, state_table, action, reward_sum


async def start_battle(ps_websocket_client, pokemon_battle_type, build_model_bool):
    if "random" in pokemon_battle_type:
        Scoring.POKEMON_ALIVE_STATIC = 30  # random battle benefits from a lower static score for an alive pkmn
        battle = await start_random_battle(ps_websocket_client, pokemon_battle_type)
    else:
        battle, model, state_table, action, reward_sum = await start_standard_battle(ps_websocket_client, pokemon_battle_type, build_model_bool)

    await ps_websocket_client.send_message(battle.battle_tag, [config.greeting_message])
    await ps_websocket_client.send_message(battle.battle_tag, ['/timer on'])

    return battle, model, state_table, action, reward_sum

model_name = "models/model.h5"
async def pokemon_battle(ps_websocket_client, pokemon_battle_type):
    if os.path.exists(model_name):
        build_model_bool = False
    else:
        print("model does not exist, initializing model")
        build_model_bool = True
        episode = 0
    battle, model, state_table, action, reward_sum = await start_battle(ps_websocket_client, pokemon_battle_type, build_model_bool)
    if os.path.exists(model_name):
        model = load_model(model_name)
        episode = data_analysis.save_or_get_episode_number("models/episodes.txt", mode = "get")
        print(f"model already exists. Number of battles played = {episode}")
    while True:
        msg = await ps_websocket_client.receive_message()
        if battle_is_finished(battle.battle_tag, msg):

            #tf.keras.callbacks.ModelCheckpoint('models/checkpoint_{epoch:04d}.h5', save_weights_only=False, period=100)


            action_required = await async_update_battle(battle, msg)
            winner = msg.split(constants.WIN_STRING)[-1].split('\n')[0].strip()
            model, reward_sum, turns = battle.battle_win_or_lose_reward(winner, model, state_table, action, reward_sum )
            model.save(model_name)
            print(f"\nFinal battle sum: {np.round(reward_sum, 2)}\n\n")
            data_analysis.save_or_get_episode_number("models/episodes.txt", mode = "save", episode = episode + 1)
            print(f"Winner: {winner}, Number of Battles: {episode+1}, \nepsilon: {np.round(0.4*0.995**(episode+1),4)}")
            #cuda.select_device(0)
            #cuda.close()
            #data_mining
            if winner == config.username: w = 1
            else: w = 0
            metrics = [[reward_sum], [turns], [w]]
            data_analysis.save_or_get_metrics(metrics, path_to_data = "models/metrics.pickle" , mode = "save", renew = False)
            metrics = data_analysis.save_or_get_metrics(metrics, path_to_data = "models/metrics.pickle" , mode = "get", renew = False)
            data_analysis.data_analysis(metrics, save_figure_path = "data_analysis/plots/reward_sum_and_turns.pdf", save_figure_bool = True)
            await ps_websocket_client.send_message(battle.battle_tag, [config.battle_ending_message])
            await ps_websocket_client.leave_battle(battle.battle_tag, save_replay=config.save_replay)
            return winner
        else:
            action_required = await async_update_battle(battle, msg);
            #print(f"{action_required and not battle.wait}")
            if action_required and not battle.wait:
                best_move, model, state_table, action, reward_sum = await async_pick_move(battle, model, state_table, action, reward_sum, episode)
                #model.save(model_name)
                print(f"sum= {np.round(reward_sum,1)}")
                await ps_websocket_client.send_message(battle.battle_tag, best_move)


























