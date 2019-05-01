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

def plot(histories, score_saliency):
    labels = ["Original", "IVModifyScoresZero", "IVModifyScoresRand", "IVnonChangingScores", "IVdecrementScore"]
    plt.figure()
    for i,history in enumerate(histories):
        plt.plot(history['rewards'][:150], label=labels[i])
    plt.xlabel("Time")
    plt.ylabel("Reward")
    plt.title("Reward Over Time")
    plt.legend()
    # plt.savefig(SAVE_DIR + 'amidar_rewards_ex.png')
    plt.show()

    plt.figure()
    for i, saliency in enumerate(score_saliency):
        plt.plot(saliency, label=labels[i])
    plt.xlabel("Time")
    plt.ylabel("Saliency on Score")
    plt.title("Saliency on Score Over Time")
    plt.legend()
    # plt.savefig(SAVE_DIR + 'amidar_scoreSaliency_ex.png')
    plt.show()

def get_score_saliency(history):
    env, model = setUp("AmidarToyboxNoFrameskip-v4", "a2c", "./models/AmidarToyboxNoFrameskip-v4/amidar4e7_a2c.model")
    concept_pixels = get_amidar_score_pixels()
    grouped_score_saliency = []

    for history in histories:
        score_saliency = []
        for i in range(150):
            #get raw saliency score
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
            score_saliency.append(np.mean(score_pixels))
        print(score_saliency)
        grouped_score_saliency.append(score_saliency)

    return grouped_score_saliency

def get_amidar_score_pixels():
    concept_pixels = []
    for x in range(90,110):
        for y in range(195,210):
            concept_pixels += [(x,y)]
    return concept_pixels

if __name__ == '__main__':
    load_dir = "./saliency_maps/movies/a2c/AmidarToyboxNoFrameskip-v4/"
    history_paths = ["default-150-amidartoyboxnoframeskip-v4-5.pkl", "IVnonChangingScores-150-amidartoyboxnoframeskip-v4-5.pkl", \
                    "IVmultModifyScoresRand-150-amidartoyboxnoframeskip-v4-5.pkl", "IVmultModifyScores-150-amidartoyboxnoframeskip-v4-5.pkl", \
                    "IVdecrementScore-150-amidartoyboxnoframeskip-v4-5.pkl"]
    histories = []
    for path in history_paths:
        history_path = load_dir + path
        with open(history_path, "rb") as output_file:
            histories.append(pickle.load(output_file))

    # score_saliency = get_score_saliency(histories)
    # filehandler = open(SAVE_DIR + 'score_saliencies.pkl', 'wb') 
    # pickle.dump(score_saliency, filehandler)
    with open(SAVE_DIR + 'score_saliencies.pkl', "rb") as output_file:
        score_saliency = pickle.load(output_file)
    plot(histories, score_saliency)
