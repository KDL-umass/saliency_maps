from saliency_maps.experiments import SAVE_DIR
from saliency_maps.visualize_atari.saliency import *
from saliency_maps.experiments.CFimportance_breakout import setUp

import numpy as np
import argparse
import pickle
import random

import matplotlib 
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def plot(histories):
    plt.figure()
    for history in histories:
        rewards = history['rewards'][:150]
        plt.plot(rewards)
    plt.xlabel("Time")
    plt.ylabel("Reward")
    plt.title("Reward Over Time")
    # plt.savefig(SAVE_DIR + 'amidar_rewards_ex_50s.png')
    plt.show()

def get_rewards(histories):
    total_rewards = []
    for history in histories:
        rewards = history['rewards']
        total_rewards += [rewards]
    return total_rewards

if __name__ == '__main__':
    load_dir = ["./saliency_maps/movies/a2c/BreakoutToyboxNoFrameskip-v4/", "./saliency_maps/movies/a2c/AmidarToyboxNoFrameskip-v4/"]
    history_paths = ["default-150-breakouttoyboxnoframeskip-v4-{}.pkl", "IVmultModifyScoresRand-150-amidartoyboxnoframeskip-v4-{}.pkl"]
    histories = []
    for j,path in enumerate(history_paths):
        paths = []
        for i in range(5,56):
            path_ = path.format(i)
            history_path = load_dir[j] + path_
            with open(history_path, "rb") as output_file:
                paths.append(pickle.load(output_file))
        histories.append(paths)

    #get rewards
    for i,history in enumerate(histories):
        # rewards = get_rewards(history)
        # filehandler = open(load_dir[i] + 'rewards_50s.pkl', 'wb') 
        # pickle.dump(rewards, filehandler)
        plot(history)
