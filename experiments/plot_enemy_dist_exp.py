from saliency_maps.experiments import SAVE_DIR
from saliency_maps.visualize_atari.saliency import *

from saliency_maps.experiments.CFimportance_breakout import setUp
from saliency_maps.experiments.CFimportance_amidar import get_concept_pixels, world_to_pixels

from baselines.common import atari_wrappers

import numpy as np
import argparse
import pickle
import random

import matplotlib 
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def plot(distances, saliency):
    labels = ['e1', 'e2', 'e3', 'e4', 'e5']
    plt.figure()
    for i in range(5):
        plt.plot(range(len(distances)), [t[i] for t in distances], label=labels[i])
        plt.fill_between(range(len(distances)), np.subtract([t[i] for t in distances], [s[i] for s in saliency]), np.add([t[i] for t in distances], [s[i] for s in saliency]), alpha=0.2)
    plt.plot(range(len(distances)), [0]*len(distances), '--', color='black', alpha=0.5)
    plt.xlabel("Time")
    plt.ylabel("Distance to Player")
    plt.title("Distance of Enemies Over Time w.r.t. Player")
    plt.legend()
    plt.savefig(SAVE_DIR + 'amidar_dist_saliency.png')
    # plt.show()

def get_dist(history, env, model):
    distances = []

    for i in range(len(history['state_json'])):
        state_json = history['state_json'][i]
        distance = []

        turtle = atari_wrappers.get_turtle(env)
        tb = turtle.toybox
        tb.write_state_json(state_json)

        enemies = state_json['enemies']
        player_index = (state_json['player']['position']['x'], state_json['player']['position']['y'])
        player_pos = world_to_pixels(player_index, tb)
        # print('player pos', player_pos)

        for enemy in enemies:
            # print(enemy)
            enemy_index = (enemy['position']['x'], enemy['position']['y'])
            enemy_pos = world_to_pixels(enemy_index, tb)
            # print('enemy pos', enemy_pos)
            distance.append(abs(player_pos[0] - enemy_pos[0]) + abs(player_pos[1] - enemy_pos[1]))

        distances += [distance]

    return distances

def get_enemy_saliency(history, env, model):
    saliency = []

    for i in range(len(history['state_json'])):
        print(i)
        state_json = history['state_json'][i]
        frame = history['color_frame'][i]

        #get enemy pixels
        turtle = atari_wrappers.get_turtle(env)
        tb = turtle.toybox
        tb.write_state_json(state_json)
        enemy_pixels = get_concept_pixels('enemies', state_json, [frame.shape[1],frame.shape[0]], tb)
        
        #get saliency for each enemy
        actor_saliency = score_frame(model, history, i, r=2, d=5, interp_func=occlude, mode='actor')
        S = np.zeros((110, 84))
        S[18:102, :] = actor_saliency
        S = imresize(actor_saliency, size=[frame.shape[0],frame.shape[1]], interp='bilinear').astype(np.float32)

        #save in right format
        saliency_i = []
        for enemy in enemy_pixels:
            saliency_enemy_i = []
            for pixels in enemy:
                saliency_enemy_i.append(S[pixels[1]][pixels[0]])
            saliency_i += [np.mean(saliency_enemy_i)]
        saliency += [saliency_i]

    return saliency

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-s', '--saliency_method', default='perturbation', type=str, help='saliency method to be used')
    args = parser.parse_args()

    load_dir = "./saliency_maps/movies/a2c/AmidarToyboxNoFrameskip-v4/perturbation/"
    history_path = load_dir + "default-250-amidartoyboxnoframeskip-v4-7.pkl"
    SAVE_DIR = SAVE_DIR + "amidar_enemy_dist_ex/{}/".format(args.saliency_method) 

    with open(history_path, "rb") as output_file:
        history_file = pickle.load(output_file)

    env, model = setUp("AmidarToyboxNoFrameskip-v4", "a2c", "./models/AmidarToyboxNoFrameskip-v4/amidar4e7_a2c.model")

    #get distances
    print("now preprocessing distances between enemies and player")
    # distances = get_dist(history_file, env, model)
    # filehandler1 = open(SAVE_DIR + 'enemy_player_distances.pkl', 'wb') 
    # pickle.dump(distances, filehandler1)

    #get saliency scores
    print("now preprocessing saliency on each enemy")
    # saliency = get_enemy_saliency(history_file, env, model)
    # filehandler1 = open(SAVE_DIR + 'saliency_on_enemies.pkl', 'wb') 
    # pickle.dump(saliency, filehandler1)

    #plot
    with open(SAVE_DIR + 'enemy_player_distances.pkl', "rb") as output_file:
        distances = pickle.load(output_file)
    with open(SAVE_DIR + 'saliency_on_enemies.pkl', "rb") as output_file:
        saliency = pickle.load(output_file)
    plot(distances, saliency)
    
