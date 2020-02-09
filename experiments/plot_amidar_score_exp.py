from saliency_maps.visualize_atari.make_movie import setUp
from saliency_maps.visualize_atari.saliency import score_frame, occlude
from saliency_maps.object_saliency.object_saliency import score_frame_by_pixels
from saliency_maps.jacobian_saliency.jacobian_saliency import get_gradients
# from saliency_maps.jacobian_saliency.jacobian_saliency import run_through_model

from saliency_maps.experiments import SAVE_DIR

import numpy as np
import argparse
import pickle
import random

from scipy.misc import imresize
from scipy.stats import pearsonr

import matplotlib 
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
# matplotlib.rcParams.update({'font.size': 14})
# plt.rcParams["font.family"] = "Times New Roman"

def plot(reward, reward_std, saliency, saliency_std, agg='mean'):
    labels = ["original", "intermittent_reset", "random_varying", "fixed", "decremented"]

    plt.figure()
    for i,r in enumerate(reward):
        plt.plot(range(len(r)), r, label=labels[i])
        plt.fill_between(range(len(r)), np.subtract(r, reward_std[i]), np.add(r, reward_std[i]), alpha=0.2)
    plt.xlabel("Time", fontsize=16)
    plt.ylabel("Reward", fontsize=16)
    # plt.title("Reward Over Time")
    # plt.legend()
    plt.savefig(SAVE_DIR + 'amidar_rewards_50s_{}.png'.format(agg))
    plt.show()

    plt.figure()
    for i, s in enumerate(saliency):
    # for i in [0,3,2,1,4]:
        plt.plot(range(len(s)), s, label=labels[i])
        # plt.fill_between(range(len(s)), np.subtract(s, saliency_std[i]), np.add(s, saliency_std[i]), alpha=0.2)
    plt.xlabel("Time", fontsize=16)
    plt.ylabel("Saliency on Score", fontsize=16)
    # plt.ylim((0, 20))
    # plt.title("Saliency on Score Over Time")
    plt.legend(fontsize=10)
    plt.savefig(SAVE_DIR + 'amidar_scoreSaliency_50s_{}.png'.format(agg))
    # plt.show()

def plot_corr_reward_saliency(reward, saliency):
    labels = ["original", "intermittent_reset", "random_varying", "fixed", "decremented"]
    colors = ['C0', 'C1', 'C2', 'C3', 'C4']
    plt.figure()
    print(len(reward), len(saliency))
    for i,r in enumerate(reward):
        # print(r, len(saliency[i]))
        if i == 0:
            continue
        plt.scatter(reward[0] - r, saliency[0] - saliency[i], label=labels[i], color=colors[i], s=4)
        corr_i = pearsonr(reward[0] - r, saliency[0] - saliency[i])
        print(labels[i], ": ", corr_i)
    plt.xlabel("Reward Difference from Original", fontsize=16)
    plt.ylabel("Saliency Difference from Original", fontsize=16)
    # plt.title("Correlation Between the Intervention Differences in Reward and Saliency")
    # plt.legend()
    plt.savefig(SAVE_DIR + 'amidar_corr_50s.png')
    plt.show()

def plot_corr_rewards(reward_mean, agg='mean'):
    labels = ["original", "intermittent_reset", "random_varying", "fixed", "decremented"]

    for i in range(1,5):
        plt.figure()
        plt.scatter(reward_mean[0], reward_mean[i])
        plt.xlabel("Reward {}".format(labels[i]))
        plt.ylabel("Reward Original")
        # plt.title("Correlation Between Original Rewards and {} Rewards".format(labels[i]))
        plt.savefig(SAVE_DIR + 'amidar_Rcorr_{}_50s_{}.png'.format(labels[i], agg))

def plot_corr_rewards_raw(histories):
    labels = ["original", "intermittent_reset", "random_varying", "fixed", "decremented"]
    orig_history = histories[0]

    for i in range(5):
        if i == 0:
            continue
        plt.figure()
        for j, history in enumerate(histories[i]):
            if len(history['rewards']) < len(orig_history[j]['rewards']):
                min_size = len(history['rewards'])
            else:
                min_size = len(orig_history[j]['rewards'])
            # min_size = len(history['rewards'])
            plt.scatter(orig_history[j]['rewards'][:min_size], history['rewards'][:min_size])
        plt.xlabel("Reward {}".format(labels[i]))
        plt.ylabel("Reward Original")
        # plt.title("Correlation Between Original Rewards and {} Rewards".format(labels[i]))
        plt.savefig(SAVE_DIR + 'amidar_Rcorr_{}_50s_raw.png'.format(labels[i]))

def get_rewards(history, num_frames=150, agg='mean'):
    grouped_rMean = []
    grouped_rVar = []
    for i,history_type in enumerate(histories):
        total_rewards = []
        for j, history in enumerate(history_type):
            rewards = history['rewards'][:num_frames]
            while len(rewards) != num_frames:
                # print("died quickly, have to append 0s {}".format(len(rewards)))
                # print(rewards[-1])
                rewards.append(rewards[-1])
                # rewards.append(0)
            total_rewards += [rewards]
        grouped_rMean += [np.mean(total_rewards, axis=0)] if agg == 'mean' else [np.median(total_rewards, axis=0)]
        grouped_rVar += [np.std(total_rewards, axis=0)]

    return grouped_rMean, grouped_rVar

def get_score_saliency(history, saliency_method='perturbation', num_frames=150, agg='mean'):
    env, model = setUp("AmidarToyboxNoFrameskip-v4", "a2c", "./models/AmidarToyboxNoFrameskip-v4/amidar4e7_a2c.model")
    concept_pixels = get_amidar_score_pixels()
    grouped_sMean = []
    grouped_sStd = []

    for history_type in histories:
        sample_saliency = []
        for history in history_type:
            score_saliency = []
            for i in range(num_frames):
                #get raw saliency score
                if len(history['color_frame']) <= i:
                    score_saliency += [0]
                    continue
                frame = history['color_frame'][i]

                if saliency_method == 'perturbation':
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
                elif saliency_method == 'object':
                    score = score_frame_by_pixels(model, history, i, concept_pixels, mode='actor')
                    score_saliency += [score]
                elif saliency_method == 'jacobian':
                    actor_saliency = get_gradients(model, history['ins'][i], mode='actor')
                    S = np.zeros((110, 84))
                    S[18:102, :] = actor_saliency[0,:,:,3]**2
                    S = imresize(actor_saliency[0,:,:,3]**2, size=[frame.shape[0],frame.shape[1]], interp='bilinear').astype(np.float32)

                    #map saliency score to score pixels
                    score_pixels = []
                    for pixels in concept_pixels:
                        score_pixels.append(S[pixels[1]][pixels[0]])
                    score_saliency += [np.mean(score_pixels)]

            print("len score saliency: ", len(score_saliency))
            sample_saliency += [score_saliency]
        print("len sample saliency: ", len(sample_saliency))
        #save column wise mean score of all samples
        grouped_sMean += [np.mean(sample_saliency, axis=0)] if agg=='mean' else [np.median(sample_saliency, axis=0)]
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

    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-s', '--saliency_method', default='perturbation', type=str, help='saliency method to be used')
    parser.add_argument('-n', '--num_frames', default=150, type=int, help='number of frames to be processed')
    parser.add_argument('-r', '--range', default=[1,51], type=list, help='range of pkl file numbering')
    parser.add_argument('-a', '--aggregation', default='mean', type=str, help='aggregation type of different samples')
    args = parser.parse_args()

    load_dir = "./saliency_maps/movies/a2c/AmidarToyboxNoFrameskip-v4/perturbation/"
    history_paths = ["default-{}-amidartoyboxnoframeskip-v4-{}.pkl", "IVmultModifyScores-{}-amidartoyboxnoframeskip-v4-{}.pkl", \
                    "IVmultModifyScoresRand-{}-amidartoyboxnoframeskip-v4-{}.pkl", "IVnonChangingScores-{}-amidartoyboxnoframeskip-v4-{}.pkl", \
                    "IVdecrementScore-{}-amidartoyboxnoframeskip-v4-{}.pkl"]
    SAVE_DIR = SAVE_DIR + "amidar_score_ex/{}/num_frames_{}/".format(args.saliency_method, args.num_frames) 

    # histories = []
    # for path in history_paths:
    #     paths = []
    #     for i in range(args.range[0], args.range[1]):
    #         path_ = path.format(args.num_frames, i)
    #         history_path = load_dir + path_
    #         with open(history_path, "rb") as output_file:
    #             paths.append(pickle.load(output_file))
    #     histories.append(paths)

    # print(len(histories), len(histories[1]))

    # #get saliency scores
    print("now getting saliency scores")
    # saliency, saliency_std = get_score_saliency(histories, saliency_method=args.saliency_method, num_frames=args.num_frames, agg=args.aggregation)
    # filehandler1 = open(SAVE_DIR + 'score_saliencies_{}_50s.pkl'.format(args.aggregation), 'wb') 
    # pickle.dump(saliency, filehandler1)
    # filehandler2 = open(SAVE_DIR + 'score_saliencies_std_50s.pkl', 'wb') 
    # pickle.dump(saliency_std, filehandler2)
    with open(SAVE_DIR + 'score_saliencies_{}_50s.pkl'.format(args.aggregation), "rb") as output_file:
        saliency = pickle.load(output_file)
    with open(SAVE_DIR + 'score_saliencies_std_50s.pkl', "rb") as output_file:
        saliency_std = pickle.load(output_file)

    # #get rewards
    print("now preprocessing rewards")
    # rewards, rewards_std = get_rewards(histories, num_frames=args.num_frames, agg=args.aggregation)
    # filehandler1 = open(SAVE_DIR + 'rewards_{}_50s.pkl'.format(args.aggregation), 'wb') 
    # pickle.dump(rewards, filehandler1)
    # filehandler2 = open(SAVE_DIR + 'rewards_std_50s.pkl', 'wb') 
    # pickle.dump(rewards_std, filehandler2)
    with open(SAVE_DIR + 'rewards_{}_50s.pkl'.format(args.aggregation), "rb") as output_file:
        rewards = pickle.load(output_file)
    with open(SAVE_DIR + 'rewards_std_50s.pkl', "rb") as output_file:
        rewards_std = pickle.load(output_file)
    # for i in range(len(rewards_std)):
    #     print(rewards_std[i][149])

    # print(saliency)
    # print(rewards)

    plot(rewards, rewards_std, saliency, saliency_std, agg=args.aggregation)
    # plot_corr_rewards(rewards, agg=args.aggregation)
    # plot_corr_rewards_raw(histories)
    plot_corr_reward_saliency(rewards, saliency)
