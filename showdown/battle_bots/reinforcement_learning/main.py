#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  4 20:02:27 2021

@author: arevell
"""


import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam


from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory





def build_model(states, actions):
    model = Sequential()
    model.add(Flatten(input_shape=(1,states)))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model





def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy,
                  nb_actions=actions, nb_steps_warmup=10, target_model_update=1e-2)
    return dqn





from showdown.battle import Battle

from ..helpers import format_decision

from showdown.engine.objects import StateMutator
from showdown.engine.select_best_move import pick_safest
from showdown.engine.select_best_move import get_payoff_matrix

import config

import logging
logger = logging.getLogger(__name__)


def prefix_opponent_move(score_lookup, prefix):
    new_score_lookup = dict()
    for k, v in score_lookup.items():
        bot_move, opponent_move = k
        new_opponent_move = "{}_{}".format(opponent_move, prefix)
        new_score_lookup[(bot_move, new_opponent_move)] = v

    return new_score_lookup


def pick_safest_move_from_battles(battles):
    all_scores = dict()
    for i, b in enumerate(battles):
        state = b.create_state()
        mutator = StateMutator(state)
        user_options, opponent_options = b.get_all_options()
        logger.debug("Searching through the state: {}".format(mutator.state))
        scores = get_payoff_matrix(mutator, user_options, opponent_options, depth = config.search_depth, prune=True)

        prefixed_scores = prefix_opponent_move(scores, str(i))
        all_scores = {**all_scores, **prefixed_scores}

    decision, payoff = pick_safest(all_scores)
    bot_choice = decision[0]
    logger.debug("Safest: {}, {}".format(bot_choice, payoff))
    return bot_choice


class BattleBot(Battle):
    def __init__(self, *args, **kwargs):
        super(BattleBot, self).__init__(*args, **kwargs)

    def find_best_move(self):
        battles = self.prepare_battles(join_moves_together=True)
        safest_move = pick_safest_move_from_battles(battles)
        return format_decision(self, safest_move)

































