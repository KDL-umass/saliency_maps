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

def plot(reward_mean, reward_std, saliency_mean, saliency_std):
    labels = ["Original", "IVModifyScoresZero", "IVModifyScoresRand", "IVnonChangingScores", "IVdecrementScore"]

    plt.figure()
    for i,reward in enumerate(reward_mean):
        plt.plot(range(len(reward)), reward, label=labels[i])
        plt.fill_between(range(len(reward)), np.subtract(reward, reward_std[i]), np.add(reward, reward_std[i]), alpha=0.2)
    plt.xlabel("Time")
    plt.ylabel("Reward")
    plt.title("Reward Over Time")
    plt.legend()
    # plt.savefig(SAVE_DIR + 'amidar_rewards_ex_50s.png')
    plt.show()

    plt.figure()
    for i, saliency in enumerate(saliency_mean):
        plt.plot(range(len(saliency)), saliency, label=labels[i])
        plt.fill_between(range(len(saliency)), np.subtract(saliency, saliency_std[i]), np.add(saliency, saliency_std[i]), alpha=0.2)
    plt.xlabel("Time")
    plt.ylabel("Saliency on Score")
    plt.title("Saliency on Score Over Time")
    plt.legend()
    # plt.savefig(SAVE_DIR + 'amidar_scoreSaliency_ex_50s.png')
    plt.show()

def get_rewards(history):
    grouped_rMean = []
    grouped_rVar = []
    for i,history_type in enumerate(histories):
        total_rewards = []
        for j, history in enumerate(history_type):
            rewards = history['rewards'][:150]
            while len(rewards) != 150:
                # print("died quickly, have to append 0s {}".format(len(rewards)))
                # print(rewards[-1])
                rewards.append(rewards[-1])
            total_rewards += [rewards]
        grouped_rMean += [np.mean(total_rewards, axis=0)]
        grouped_rVar += [np.std(total_rewards, axis=0)]

    return grouped_rMean, grouped_rVar

def get_score_saliency(history):
    env, model = setUp("AmidarToyboxNoFrameskip-v4", "a2c", "./models/AmidarToyboxNoFrameskip-v4/amidar4e7_a2c.model")
    concept_pixels = get_amidar_score_pixels()
    grouped_sMean = []
    grouped_sStd = []

    for history_type in histories:
        sample_saliency = []
        for history in history_type:
            score_saliency = []
            for i in range(150):
                #get raw saliency score
                if len(history['color_frame']) <= i:
                    score_saliency += [score_saliency[-1]]
                    continue
                frame = history['color_frame'][i]
                actor_saliency = score_frame(model, history, i, r=2, d=5, interp_func=occlude, mode='actor')
                S = np.zeros((110, 84))
                S[18:102, :] = actor_saliency
                S = imresize(actor_saliency, size=[frame.shape[0],frame.shape[1]], interp='bilinear').astype(np.float32)

                #change pixels to white to see mapping in the real frame
                # for pixel in concept_pixels:
                #     frame[pixel[1], pixel[0], 0] = 255
                #     frame[pixel[1], pixel[0], 1] = 255
                #     frame[pixel[1], pixel[0], 2] = 255
                # plt.imshow(frame)
                # plt.show()

                #map saliency score to score pixels
                score_pixels = []
                for pixels in concept_pixels:
                    score_pixels.append(S[pixels[1]][pixels[0]])
                score_saliency += [np.mean(score_pixels)]
            print("len score saliency: ", len(score_saliency))
            sample_saliency += [score_saliency]
        print("len sample saliency: ", len(sample_saliency))
        #save column wise mean score of all samples
        grouped_sMean += [np.mean(sample_saliency, axis=0)]
        grouped_sStd += [np.std(sample_saliency, axis=0)]
        print(grouped_sMean)
        print(grouped_sStd)

    return grouped_sMean, grouped_sStd

def get_amidar_score_pixels():
    concept_pixels = []
    for x in range(80,105):
        for y in range(195,210):
            concept_pixels += [(x,y)]
    return concept_pixels

if __name__ == '__main__':
    load_dir = "./saliency_maps/movies/a2c/AmidarToyboxNoFrameskip-v4/"
    history_paths = ["default-150-amidartoyboxnoframeskip-v4-{}.pkl", "IVnonChangingScores-150-amidartoyboxnoframeskip-v4-{}.pkl", \
                    "IVmultModifyScoresRand-150-amidartoyboxnoframeskip-v4-{}.pkl", "IVmultModifyScores-150-amidartoyboxnoframeskip-v4-{}.pkl", \
                    "IVdecrementScore-150-amidartoyboxnoframeskip-v4-{}.pkl"]
    SAVE_DIR = SAVE_DIR + "amidar_score_ex/" 

    histories = []
    for path in history_paths:
        paths = []
        for i in range(5,55):
            path_ = path.format(i)
            history_path = load_dir + path_
            with open(history_path, "rb") as output_file:
                paths.append(pickle.load(output_file))
        histories.append(paths)

    print(len(histories), len(histories[1]))

    # #get saliency scores
    print("now getting saliency scores")
    # saliency_mean, saliency_std = get_score_saliency(histories)
    # filehandler1 = open(SAVE_DIR + 'score_saliencies_mean_50s.pkl', 'wb') 
    # pickle.dump(saliency_mean, filehandler1)
    # filehandler2 = open(SAVE_DIR + 'score_saliencies_std_50s.pkl', 'wb') 
    # pickle.dump(saliency_std, filehandler2)
    with open(SAVE_DIR + 'score_saliencies_mean_50s.pkl', "rb") as output_file:
        saliency_mean = pickle.load(output_file)
    with open(SAVE_DIR + 'score_saliencies_std_50s.pkl', "rb") as output_file:
        saliency_std = pickle.load(output_file)

    # #get rewards
    print("now preprocessing rewards")
    # rewards_mean, rewards_std = get_rewards(histories)
    # filehandler1 = open(SAVE_DIR + 'rewards_mean_50s.pkl', 'wb') 
    # pickle.dump(rewards_mean, filehandler1)
    # filehandler2 = open(SAVE_DIR + 'rewards_std_50s.pkl', 'wb') 
    # pickle.dump(rewards_std, filehandler2)
    with open(SAVE_DIR + 'rewards_mean_50s.pkl', "rb") as output_file:
        rewards_mean = pickle.load(output_file)
    with open(SAVE_DIR + 'rewards_std_50s.pkl', "rb") as output_file:
        rewards_std = pickle.load(output_file)
    # for i in range(len(rewards_std)):
    #     print(rewards_std[i][149])

    plot(rewards_mean, rewards_std, saliency_mean, saliency_std)
