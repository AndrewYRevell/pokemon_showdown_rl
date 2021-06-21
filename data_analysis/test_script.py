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

from environs import Env
from collections import defaultdict

import data_analysis.config as config
import config
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


#%%

def parse_configs():
    env = Env()
    env.read_env("./data_analysis")

    config.battle_bot_module = env("BATTLE_BOT", 'safest')
    config.save_replay = env.bool("SAVE_REPLAY", config.save_replay)
    config.use_relative_weights = env.bool("USE_RELATIVE_WEIGHTS", config.use_relative_weights)
    config.gambit_exe_path = env("GAMBIT_PATH", config.gambit_exe_path)
    config.search_depth = int(env("MAX_SEARCH_DEPTH", config.search_depth))
    config.greeting_message = env("GREETING_MESSAGE", config.greeting_message)
    config.battle_ending_message = env("BATTLE_OVER_MESSAGE", config.battle_ending_message)
    config.websocket_uri = env("WEBSOCKET_URI", "sim.smogon.com:8000")
    config.pokemon_mode = env("POKEMON_MODE", constants.DEFAULT_MODE)
    config.run_count = int(env("RUN_COUNT", 1))
    config.room_name = env("ROOM_NAME", config.room_name)
    init_logging(env("LOG_LEVEL", "DEBUG"))
    
    config.username1 = env("PS_USERNAME1")
    config.password1 = env("PS_PASSWORD1", "")
    config.bot_mode1 = env("BOT_MODE1")
    config.team_name1 = env("TEAM_NAME1", None)
    config.user_to_challenge1 = env("USER_TO_CHALLENGE1", config.user_to_challenge1)
    config.username2 = env("PS_USERNAME2")
    config.password2 = env("PS_PASSWORD2", "")
    config.bot_mode2 = env("BOT_MODE2")
    config.team_name2 = env("TEAM_NAME2", None)
    config.user_to_challenge2 = env("USER_TO_CHALLENGE2", config.user_to_challenge2)
    
parse_configs()
apply_mods(config.pokemon_mode)



async def showdown():
    ps_websocket_client1 = await PSWebsocketClient.create(config.username1, config.password1, config.websocket_uri)
    await ps_websocket_client1.login()
    
    ps_websocket_client2 = await PSWebsocketClient.create(config.username2, config.password2, config.websocket_uri)
    await ps_websocket_client2.login()
    
    team1 = load_team(config.team_name1)
    team2 = load_team(config.team_name2)
    
    await ps_websocket_client1.challenge_user(config.user_to_challenge1, config.pokemon_mode, team1)
    await ps_websocket_client2.accept_challenge(config.pokemon_mode, team2, config.room_name, config.user_to_challenge2)


    await asyncio.gather(     
        pokemon_battle(ps_websocket_client1, config.pokemon_mode),
        pokemon_battle(ps_websocket_client2, config.pokemon_mode)
    )
    

#%%

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
    config.pokemon_mode = env("POKEMON_MODE", constants.DEFAULT_MODE)
    config.run_count = int(env("RUN_COUNT", 1))
    config.room_name = env("ROOM_NAME", config.room_name)
    init_logging(env("LOG_LEVEL", "DEBUG"))
    
    config.username = env("PS_USERNAME")
    config.password = env("PS_PASSWORD", "")
    config.bot_mode = env("BOT_MODE")
    config.team_name = env("TEAM_NAME", None)
    config.user_to_challenge = env("USER_TO_CHALLENGE", config.user_to_challenge)
    
parse_configs()
apply_mods(config.pokemon_mode)


async def showdown():
    ps_websocket_client = await PSWebsocketClient.create(config.username, config.password, config.websocket_uri)
    await ps_websocket_client.login()
    
    team = load_team(config.team_name)

    
    await ps_websocket_client.challenge_user(config.user_to_challenge, config.pokemon_mode, team)


    
    await pokemon_battle(ps_websocket_client, config.pokemon_mode),



#%%
    
showdown()

load_team(config.team_name1)


p1_side_init = Side(
                Pokemon.from_state_pokemon_dict(StatePokemon("raichu", 73).to_dict()),
                {
                    "xatu": Pokemon.from_state_pokemon_dict(StatePokemon("xatu", 81).to_dict()),
                    "starmie": Pokemon.from_state_pokemon_dict(StatePokemon("starmie", 81).to_dict()),
                    "gyarados": Pokemon.from_state_pokemon_dict(StatePokemon("gyarados", 81).to_dict()),
                    "dragonite": Pokemon.from_state_pokemon_dict(StatePokemon("dragonite", 81).to_dict()),
                    "hitmonlee": Pokemon.from_state_pokemon_dict(StatePokemon("hitmonlee", 81).to_dict()),
                },
                (0, 0),
                defaultdict(lambda: 0)
            )

p2_side_init = Side(
                Pokemon.from_state_pokemon_dict(StatePokemon("aromatisse", 81).to_dict()),
                {
                    "yveltal": Pokemon.from_state_pokemon_dict(StatePokemon("yveltal", 73).to_dict()),
                    "slurpuff": Pokemon.from_state_pokemon_dict(StatePokemon("slurpuff", 73).to_dict()),
                    "victini": Pokemon.from_state_pokemon_dict(StatePokemon("victini", 73).to_dict()),
                    "toxapex": Pokemon.from_state_pokemon_dict(StatePokemon("toxapex", 73).to_dict()),
                    "bronzong": Pokemon.from_state_pokemon_dict(StatePokemon("bronzong", 73).to_dict()),
                },
                (0, 0),
                defaultdict(lambda: 0)
            )


p1_state_init = State( p1_side_init, p2_side_init, None, None, False )
p1_state_init = State( p2_side_init, p1_side_init, None, None, False )




























