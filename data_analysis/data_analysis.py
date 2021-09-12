#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 08:51:00 2021

@author: arevell
"""

import os
import pickle
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import seaborn as sns

#%
def save_pickle(data, fname):
    with open(fname, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(data, output, pickle.HIGHEST_PROTOCOL)

def open_pickle(fname):
    with open(fname, 'rb') as f: data = pickle.load(f)
    return data

def save_or_get_episode_number(path_to_episode_txt_file, mode = "get", episode = 0):
    if mode == "get":
        if os.path.isfile(path_to_episode_txt_file):
            with open(path_to_episode_txt_file) as f:
                episodes = f.readlines()
            episodes = int(episodes[0])
        else:
            print("Can't get episode number because file does not exist. Setting episode number to zero")
            episodes = 0
        return episodes
    if mode == "save":
        with open(path_to_episode_txt_file, 'w') as f:
            f.write(str(episode))


def save_or_get_metrics(metrics = None, path_to_data = "data/metrics.pickle" , mode = "get", renew = False):
    if mode == "get":
        if os.path.isfile(path_to_data): metrics = open_pickle(path_to_data)
        else: print("file does not exist.")
        return metrics
    if mode == "save":
        if os.path.isfile(path_to_data):
            metrics_old = open_pickle(path_to_data)
            for i in range(len(metrics_old)):
                metrics_old[i].extend(metrics[i]    )
            metrics = metrics_old
        save_pickle(metrics, path_to_data)


def plot_make(r = 1, c = 1, size_length = None, size_height = None, dpi = 300, sharex = False, sharey = False , squeeze = True):
    if size_length == None:
       size_length = 4* c
    if size_height == None:
        size_height = 4 * r
    fig, axes = plt.subplots(r, c, figsize=(size_length, size_height), dpi=dpi, sharex =sharex, sharey = sharey, squeeze = squeeze)
    return fig, axes


def save_figure(fname, save_figure = True, bbox_inches = None, pad_inches = 0.1):
    #allows option to not save figures when saveFigures == False
    if save_figure == True:
        plt.savefig(fname, bbox_inches=bbox_inches, pad_inches = pad_inches)





#%%
from scipy import interpolate
from scipy.ndimage.filters import gaussian_filter1d

from scipy.interpolate import make_interp_spline, BSpline

def data_analysis(metrics, save_figure_path = None, save_figure_bool = True, rolling_average = 50):
    """
    save_figure_path = "data_analysis/plots/reward_sum_and_turns.pdf"
    metrics = save_or_get_metrics(path_to_data = "models/metrics.pickle" , mode = "get", renew = False)
    """
    reward_sum, turns, wins = metrics

    #rolling average wins
    df = pd.DataFrame(wins, columns = ["win or lose"])
    if len(wins) < rolling_average:
        rolling_average = 2
    df_rolling = df["win or lose"].rolling( rolling_average, center = False).mean()



    fig, axes = plot_make(size_length = 6, r=3)
    sns.scatterplot(data = df, ax = axes[0], s=10,  linewidth=0, palette = ["#11418199"])

    fig.text(0.5, 0.92, 'Pokemon Showdown Reinforcement Learning Easy Team', horizontalalignment='center', verticalalignment='center', size = 14)


    sns.lineplot(data = df_rolling, ax = axes[0],  color = "#114181", linewidth = 3)
    axes[0].spines['top'].set_visible(False); axes[0].spines['right'].set_visible(False)
    axes[0].set_title("Wins (1) or Loses (0)")
    axes[0].get_legend().remove()

    sns.scatterplot(data = reward_sum, ax = axes[1], color = "#1c6dd8", s=10,  linewidth=0)
    axes[1].spines['top'].set_visible(False); axes[1].spines['right'].set_visible(False)
    axes[1].axhline(0, color = "#444444", ls = "--")
    axes[1].set_ylabel("Reward" )
    axes[1].set_title("Cumulative Reward from a Battle")
    axes[1].set_ylim([-15,815])

    ysmoothed = gaussian_filter1d(reward_sum,  sigma=50)
    sns.lineplot(data=  ysmoothed, ax =  axes[1], color = "#1c6dd8", linewidth = 5)


    sns.scatterplot(data = turns, ax = axes[2], color = "#8dafdd", s=10,  linewidth=0)
    axes[2].set_ylabel('turns')
    axes[2].spines['top'].set_visible(False); axes[2].spines['right'].set_visible(False)
    ysmoothed = gaussian_filter1d(turns,  sigma=50)
    sns.lineplot(data=  ysmoothed, ax =  axes[2], color = "#8dafdd", linewidth = 5)
    axes[2].set_title("Number of Turns Per Battle")

    axes[2].set_xlabel("battle number")

    if save_figure_bool:
        if not save_figure_path == None:
            save_figure(save_figure_path, save_figure = save_figure_bool, bbox_inches = None, pad_inches = 0.1)
    #plt.show()
    plt.clf()






















