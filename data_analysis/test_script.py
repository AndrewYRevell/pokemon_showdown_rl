#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 08:51:00 2021

@author: arevell
"""

import os
import logging
import unittest

from environs import Env
from collections import defaultdict

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




env = Env()
env.read_env()

load_team(config.team_name)

logger = logging.getLogger(__name__)


def parse_configs():
    env = Env()
    env.read_env("./environment.txt")

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
    config.user_to_challenge = env("USER_TO_CHALLENGE", config.user_to_challenge)
    init_logging(env("LOG_LEVEL", "DEBUG"))



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




























