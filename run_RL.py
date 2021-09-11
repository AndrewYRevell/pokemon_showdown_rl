#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 08:51:00 2021

@author: arevell
"""

import os
import logging
import unittest
import asyncio
from copy import deepcopy
import json

from environs import Env
from collections import defaultdict

import data_analysis.config as config
import constants

from teams import load_team
from config import init_logging
from data import all_move_json
from data import pokedex
from data.mods.apply_mods import apply_mods

from showdown.engine.objects import State
from showdown.engine.objects import Side
from showdown.engine.objects import Pokemon
from showdown.battle import Pokemon as StatePokemon
from showdown.websocket_client import PSWebsocketClient
from showdown.run_battle import pokemon_battle

logger = logging.getLogger(__name__)

from numba import cuda
import time
#%

import config
from showdown.battle_bots.helpers import format_decision

def parse_configs():
    env = Env()
    env.read_env()

    config.battle_bot_module = env("BATTLE_BOT", 'safest')
    config.save_replay = env.bool("SAVE_REPLAY", config.save_replay)
    config.use_relative_weights = env.bool("USE_RELATIVE_WEIGHTS", config.use_relative_weights)
    config.gambit_exe_path = env("GAMBIT_PATH", config.gambit_exe_path)
    config.search_depth = int(env("MAX_SEARCH_DEPTH", config.search_depth))
    config.greeting_message = env("GREETING_MESSAGE", config.greeting_message)
    config.battle_ending_message = env("BATTLE_OVER_MESSAGE", config.battle_ending_message)
    config.websocket_uri = env("WEBSOCKET_URI", "sim.smogon.com:8000")
    config.username = env("PS_USERNAME")
    config.password = env("PS_PASSWORD", "")
    config.bot_mode = env("BOT_MODE")
    config.team_name = env("TEAM_NAME", None)
    config.pokemon_mode = env("POKEMON_MODE", constants.DEFAULT_MODE)
    config.run_count = int(env("RUN_COUNT", 1))
    config.room_name = env("ROOM_NAME", config.room_name)

    if config.bot_mode == constants.CHALLENGE_USER:
        config.user_to_challenge = env("USER_TO_CHALLENGE")
    if config.bot_mode == constants.ACCEPT_CHALLENGE:
        config.user_to_challenge = env("USER_TO_CHALLENGE")
    init_logging(env("LOG_LEVEL", "DEBUG"))


def check_dictionaries_are_unmodified(original_pokedex, original_move_json):
    # The bot should not modify the data dictionaries
    # This is a "just-in-case" check to make sure and will stop the bot if it mutates either of them
    tmp = 0
    """
    if original_move_json != all_move_json:
        logger.critical("Move JSON changed!\nDumping modified version to `modified_moves.json`")
        with open("modified_moves.json", 'w') as f:
            json.dump(all_move_json, f, indent=4)
        exit(1)
    else:
        logger.debug("Move JSON unmodified!")

    if original_pokedex != pokedex:
        logger.critical("Pokedex JSON changed!\nDumping modified version to `modified_pokedex.json`")
        with open("modified_pokedex.json", 'w') as f:
            json.dump(pokedex, f, indent=4)
        exit(1)
    else:
        logger.debug("Pokedex JSON unmodified!")
    """

#%%
#config.run_count = 200
async def showdown():
    parse_configs()
    apply_mods(config.pokemon_mode)

    original_pokedex = deepcopy(pokedex)
    original_move_json = deepcopy(all_move_json)

    battles_run = 0
    wins = 0
    losses = 0

    t00 = time.time()

    ps_websocket_client = await PSWebsocketClient.create(config.username, config.password, config.websocket_uri)
    await ps_websocket_client.login()
    while True:
        t0 = time.time()
        team = load_team(config.team_name)
        time.sleep(3)
        pokemon_battle_type = config.pokemon_mode

        await ps_websocket_client.challenge_user(config.user_to_challenge, config.pokemon_mode, team)




        winner = await pokemon_battle(ps_websocket_client, config.pokemon_mode)
        if winner == config.username:
            wins += 1
        else:
            losses += 1
        battles_run += 1
        t = calculate_elapsed_time(t0, time.time())
        print(f"\n\nElapsed Time: {t}\nwins: {wins}, losses: {losses}, total: {wins + losses}/{config.run_count}\n\n")

        check_dictionaries_are_unmodified(original_pokedex, original_move_json)
        if battles_run >= config.run_count:
            print(f"\nElapsed Time for Batch: {calculate_elapsed_time(t00, time.time())}\n\n\n\n\n")
            break

def calculate_elapsed_time(t0, t1):
    duration = t1-t0
    mapping = [
        ('s', 60),
        ('m', 60),
        ('h', 24),
    ]
    duration = int(duration)
    result = []
    for symbol, max_amount in mapping:
        amount = duration % max_amount
        result.append(f'{amount}{symbol}')
        duration //= max_amount
        if duration == 0:
            break

    if duration:
        result.append(f'{duration}d')

    return ' '.join(reversed(result))



if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(showdown())












