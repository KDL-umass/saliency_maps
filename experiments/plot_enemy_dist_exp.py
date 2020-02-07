from saliency_maps.experiments import SAVE_DIR
from saliency_maps.visualize_atari.saliency import *

from saliency_maps.experiments.CFimportance_breakout import setUp
from saliency_maps.utils.get_concept_pixels import get_concept_pixels_amidar, world_to_pixels
from saliency_maps.object_saliency.object_saliency import score_frame_by_pixels
from saliency_maps.jacobian_saliency.jacobian_saliency import get_gradients

from baselines.common import atari_wrappers

import numpy as np
import argparse
import pickle
import random

from scipy.stats import pearsonr
from scipy.stats import ttest_rel
from scipy.stats import linregress

import matplotlib 
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def plot_icorrelation(i_distance, i_saliency, individual=False):
    labels = ['e1', 'e2', 'e3', 'e4', 'e5']
    colors = ['C0', 'C3', 'C2', 'C1', 'C4']
    plt.figure()

    if individual:
        for i in range(len(i_distance)):
            for j in [-4,-2,2,4]:
                # if j != 0:
                #     plt.scatter([j]*len(i_saliency[i][j]), i_saliency[i][j], label=labels[i], color=colors[i])
                # else:
                # print(i_saliency[0][j])
                # print(i_saliency[i][j])
                plt.scatter([j]*len(i_saliency[i][j]), np.subtract(i_saliency[i][0], i_saliency[i][j]), label=labels[i], color=colors[i], alpha=0.5)
            # plt.legend()
            # plt.show()
            plt.title("Intervention on Enemy {}'s Location w.r.t. Player".format(i+1))
            plt.xlabel("Intervention on Distance to Player")
            plt.ylabel("Difference of Saliency on Enemy {}".format(i+1))
            plt.xticks([-4,-2,0,2,4])
            plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
            plt.plot(range(-4,5), [0]*len(range(-4,5)), '--', color='black', alpha=0.5)
            # plt.ylim(40,-40)
            plt.savefig(SAVE_DIR + 'amidar_dist_intervention_e{}.png'.format(i+1))
            plt.clf()
    else:
        for i in range(len(i_distance)):
            enemyi_x = []
            enemyi_y = []
            for j in [-4,-2,2,4]:
                # if j != 0:
                #     plt.scatter([j]*len(i_saliency[i][j]), i_saliency[i][j], label=labels[i], color=colors[i])
                # else:
                # print(i_saliency[0][j])
                # print(i_saliency[i][j])
                plt.scatter([j]*len(i_saliency[i][j]), np.subtract(i_saliency[i][0], i_saliency[i][j]), label=labels[i], color=colors[i], alpha=0.2)
                enemyi_x += [j]*len(i_saliency[i][j])
                enemyi_y += np.subtract(i_saliency[i][0], i_saliency[i][j]).tolist()

            # slope, intercept = np.polyfit([-4,-2,2,4], enemyi_y, 1)
            # plt.plot([-4,-2,2,4], intercept + np.multiply(slope,enemyi_y), '-', color=colors[i])
            # print(slope, intercept)
            # print(len(enemyi_x), len(enemyi_y))
            slope, intercept, r_value, p_value, std_err = linregress(enemyi_x, enemyi_y)
            print('Interventional Regression Results (slope, intercept, r_value, p_value, std_err)')
            print('-------enemy {}-------'.format(i+1))
            print(slope, intercept, r_value, p_value, std_err)
            plt.plot(enemyi_x, intercept + np.multiply(slope,enemyi_x), '-', color=colors[i])

        # plt.legend()
        # plt.show()
        plt.title("Intervention on Enemy Location w.r.t. Player")
        plt.xlabel("Intervention on Distance to Player")
        plt.ylabel("Difference of Saliency on Enemy")
        plt.xticks([-4,-2,0,2,4])
        plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        plt.plot(range(-4,5), [0]*len(range(-4,5)), '--', color='black', alpha=0.5)
        # plt.ylim(40,-40)
        plt.savefig(SAVE_DIR + 'amidar_dist_intervention_bestFit.png')
        plt.show()

def plot(distances, saliency):
    labels = ['e1', 'e2', 'e3', 'e4', 'e5']
    plt.figure()
    for i in range(5):
        plt.plot(range(len(distances)), [t[i] for t in distances], label=labels[i])
        plt.fill_between(range(len(distances)), np.subtract([t[i] for t in distances], [s[i] for s in saliency]), np.add([t[i] for t in distances], [s[i] for s in saliency]), alpha=0.2)
    plt.plot(range(len(distances)), [0]*len(distances), '--', color='black', alpha=0.8)
    plt.xlabel("Time")
    plt.ylabel("Distance to Player")
    plt.title("Distance of Enemies w.r.t. Player Over Time")
    plt.legend()
    plt.savefig(SAVE_DIR + 'amidar_dist_saliency_time.png')
    # plt.show()

def plot_correlation(distances, saliency, individual=False):
    labels = ['e1', 'e2', 'e3', 'e4', 'e5']
    colors = ['C0', 'C3', 'C2', 'C1', 'C4']
    plt.figure()
    if individual:
        for i in range(5):
            plt.scatter([t[i] for t in distances], [s[i] for s in saliency], label=labels[i], color=colors[i], alpha=0.5)
            plt.ylabel("Saliency on Enemy {}".format(i+1))
            plt.xlabel("Distance to Player")
            plt.title("Observed Distance of Enemy {} w.r.t. Player".format(i+1))
            # plt.legend()
            plt.savefig(SAVE_DIR + 'amidar_dist_saliency_correlation_e{}.png'.format(i+1))
            plt.clf()
    else:
        for i in range(5):
            plt.scatter([t[i] for t in distances], [s[i] for s in saliency], label=labels[i], color=colors[i], alpha=0.2)
            # slope, intercept = np.polyfit([t[i] for t in distances], [s[i] for s in saliency], 1)
            # plt.plot([t[i] for t in distances], intercept + np.multiply(slope,[t[i] for t in distances]), '-', color=colors[i])
            # print(slope, intercept)
            slope, intercept, r_value, p_value, std_err = linregress([t[i] for t in distances], [s[i] for s in saliency])
            print('Observed Correlation Results (slope, intercept, r_value, p_value, std_err)')
            print('-------enemy {}-------'.format(i+1))
            print(slope, intercept, r_value, p_value, std_err)
            plt.plot([t[i] for t in distances], intercept + np.multiply(slope,[t[i] for t in distances]), '-', color=colors[i])
        plt.ylabel("Saliency on Enemy")
        plt.xlabel("Distance to Player")
        plt.title("Observed Enemy Distance w.r.t. Player")
        plt.legend()
        # plt.savefig(SAVE_DIR + 'amidar_dist_saliency_correlation.png')
        plt.savefig(SAVE_DIR + 'amidar_dist_saliency_corrBestFit.png')
        plt.show()

def get_pearson_correlation(distances, saliency):
    for i in range(5):
        corr = pearsonr([t[i] for t in distances], [s[i] for s in saliency])
        print("Enemy {}: ".format(i+1), corr)

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

def get_enemy_saliency(history, env, model, saliency_method):
    saliency = []

    for i in range(len(history['state_json'])):
        print(i)
        state_json = history['state_json'][i]
        frame = history['color_frame'][i]

        #get enemy pixels
        turtle = atari_wrappers.get_turtle(env)
        tb = turtle.toybox
        tb.write_state_json(state_json)
        enemy_pixels = get_concept_pixels_amidar('enemies', state_json, [frame.shape[1],frame.shape[0]], tb)
        
        #get saliency for each enemy
        #save in right format
        saliency_i = []
        if saliency_method == 'perturbation':
            actor_saliency = score_frame(model, history, i, r=2, d=5, interp_func=occlude, mode='actor')
            S = np.zeros((110, 84))
            S[18:102, :] = actor_saliency
            S = imresize(actor_saliency, size=[frame.shape[0],frame.shape[1]], interp='bilinear').astype(np.float32)

            for enemy in enemy_pixels:
                saliency_enemy_i = []
                for pixels in enemy:
                    saliency_enemy_i.append(S[pixels[1]][pixels[0]])
                saliency_i += [np.mean(saliency_enemy_i)]
            saliency += [saliency_i]
        elif saliency_method == 'object':
            for enemy in enemy_pixels:
                saliency_enemy_i = score_frame_by_pixels(model, history, i, enemy, mode='actor')
                saliency_i += [saliency_enemy_i]
            saliency += [saliency_i]
        elif saliency_method == 'jacobian':
            actor_saliency = get_gradients(model, history['ins'][i], mode='actor')
            S = np.zeros((110, 84))
            S[18:102, :] = actor_saliency[0,:,:,3]**2
            S = imresize(actor_saliency[0,:,:,3]**2, size=[frame.shape[0],frame.shape[1]], interp='bilinear').astype(np.float32)

            for enemy in enemy_pixels:
                saliency_enemy_i = []
                for pixels in enemy:
                    saliency_enemy_i.append(S[pixels[1]][pixels[0]])
                saliency_i += [np.mean(saliency_enemy_i)]
            print(saliency_i)
            saliency += [saliency_i]

    return saliency

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-s', '--saliency_method', default='perturbation', type=str, help='saliency method to be used')
    args = parser.parse_args()

    load_dir = "./saliency_maps/movies/a2c/AmidarToyboxNoFrameskip-v4/perturbation/"
    # history_path = load_dir + "default-250-amidartoyboxnoframeskip-v4-7.pkl"
    history_path = load_dir + "default-1000-amidartoyboxnoframeskip-v4-6.pkl"
    SAVE_DIR = SAVE_DIR + "amidar_enemy_dist_ex/{}/".format(args.saliency_method) 

    with open(history_path, "rb") as output_file:
        history_file = pickle.load(output_file)

    env, model = setUp("AmidarToyboxNoFrameskip-v4", "a2c", "./models/AmidarToyboxNoFrameskip-v4/amidar4e7_a2c.model")

    #get distances
    # print("now preprocessing distances between enemies and player")
    # distances = get_dist(history_file, env, model)
    # filehandler1 = open(SAVE_DIR + 'enemy_player_distances.pkl', 'wb') 
    # pickle.dump(distances, filehandler1)

    # # #get saliency scores
    # print("now preprocessing saliency on each enemy")
    # saliency = get_enemy_saliency(history_file, env, model, args.saliency_method)
    # filehandler1 = open(SAVE_DIR + 'saliency_on_enemies.pkl', 'wb') 
    # pickle.dump(saliency, filehandler1)

    with open(SAVE_DIR + 'enemy_player_distances.pkl', "rb") as output_file:
        distances = pickle.load(output_file)
    with open(SAVE_DIR + 'saliency_on_enemies.pkl', "rb") as output_file:
        saliency = pickle.load(output_file)
    # plot(distances, saliency)
    plot_correlation(distances, saliency)#, individual=True)
    # get_pearson_correlation(distances, saliency)

    #plot intervention
    with open(SAVE_DIR + 'intervention_distances.pkl', "rb") as output_file:
        intervention_distances = pickle.load(output_file)
    with open(SAVE_DIR + 'intervention_saliency.pkl', "rb") as output_file:
        intervention_saliency = pickle.load(output_file)

    plot_icorrelation(intervention_distances, intervention_saliency)#, individual=True)
    # get_ttest_icorrelation(intervention_distances, intervention_saliency)
    
