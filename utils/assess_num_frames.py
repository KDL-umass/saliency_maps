from saliency_maps.visualize_atari.saliency import score_frame
from saliency_maps.object_saliency.object_saliency import score_frame_by_pixels
# from saliency_maps.jacobian_saliency.jacobian_saliency import run_through_model

from saliency_maps.experiments import SAVE_DIR
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
    plt.savefig(SAVE_DIR + 'amidar_rewards_ex_50s.png')
    # plt.show()

    plt.figure()
    for i, saliency in enumerate(saliency_mean):
        plt.plot(range(len(saliency)), saliency, label=labels[i])
        plt.fill_between(range(len(saliency)), np.subtract(saliency, saliency_std[i]), np.add(saliency, saliency_std[i]), alpha=0.2)
    plt.xlabel("Time")
    plt.ylabel("Saliency on Score")
    plt.title("Saliency on Score Over Time")
    plt.legend()
    plt.savefig(SAVE_DIR + 'amidar_scoreSaliency_ex_50s.png')
    # plt.show()

def plot_correlation(reward_mean):
    labels = ["Original", "IVModifyScoresZero", "IVModifyScoresRand", "IVnonChangingScores", "IVdecrementScore"]

    for i in range(1,5):
        plt.figure()
        plt.scatter(reward_mean[0], reward_mean[i])
        plt.xlabel("Reward {}".format(labels[i]))
        plt.ylabel("Reward Original")
        plt.title("Correlation Between Original Rewards and {} Rewards".format(labels[i]))
        plt.savefig(SAVE_DIR + 'amidar_Rcorr_{}_50s.png'.format(labels[i]))

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
                    score_saliency += [score_saliency[-1]]
                    continue
                frame = history['color_frame'][i]
                obs = history['color_frame'][i-1]

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
                    saliency = run_through_model(model, obs)
                    S = np.zeros((110, 84))
                    S[18:102, :] = saliency
                    S = imresize(saliency, size=[frame.shape[0],frame.shape[1]], interp='bilinear').astype(np.float32)

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
    parser.add_argument('-a', '--aggregation', default='median', type=str, help='aggregation type of different samples')
    args = parser.parse_args()

    load_dir = "./saliency_maps/movies/a2c/AmidarToyboxNoFrameskip-v4/perturbation/"
    history_paths = ["default-{}-amidartoyboxnoframeskip-v4-{}.pkl", "IVnonChangingScores-{}-amidartoyboxnoframeskip-v4-{}.pkl", \
                    "IVmultModifyScoresRand-{}-amidartoyboxnoframeskip-v4-{}.pkl", "IVmultModifyScores-{}-amidartoyboxnoframeskip-v4-{}.pkl", \
                    "IVdecrementScore-{}-amidartoyboxnoframeskip-v4-{}.pkl"]
    SAVE_DIR = SAVE_DIR + "amidar_score_ex/{}/num_frames_{}/".format(args.saliency_method, args.num_frames) 

    histories = []
    for path in history_paths:
        paths = []
        for i in range(args.range[0], args.range[1]):
            path_ = path.format(args.num_frames, i)
            history_path = load_dir + path_
            with open(history_path, "rb") as output_file:
                paths.append(pickle.load(output_file))
        histories.append(paths)

    print(len(histories), len(histories[1]))

    # #get saliency scores
    print("now getting saliency scores")
    # saliency_mean, saliency_std = get_score_saliency(histories, saliency_method=args.saliency_method, num_frames=args.num_frames, agg=args.aggregation)
    # filehandler1 = open(SAVE_DIR + 'score_saliencies_{}_50s.pkl'.format(args.aggregation), 'wb') 
    # pickle.dump(saliency_mean, filehandler1)
    # filehandler2 = open(SAVE_DIR + 'score_saliencies_std_50s.pkl', 'wb') 
    # pickle.dump(saliency_std, filehandler2)
    # with open(SAVE_DIR + 'score_saliencies_{}_50s.pkl'.format(args.aggregation), "rb") as output_file:
    #     saliency_mean = pickle.load(output_file)
    # with open(SAVE_DIR + 'score_saliencies_std_50s.pkl', "rb") as output_file:
    #     saliency_std = pickle.load(output_file)

    # #get rewards
    print("now preprocessing rewards")
    rewards_mean, rewards_std = get_rewards(histories, num_frames=args.num_frames, agg=args.aggregation)
    filehandler1 = open(SAVE_DIR + 'rewards_{}_50s.pkl'.format(args.aggregation), 'wb') 
    pickle.dump(rewards_mean, filehandler1)
    filehandler2 = open(SAVE_DIR + 'rewards_std_50s.pkl', 'wb') 
    pickle.dump(rewards_std, filehandler2)
    # with open(SAVE_DIR + 'rewards_{}_50s.pkl'.format(args.aggregation), "rb") as output_file:
    #     rewards_mean = pickle.load(output_file)
    # with open(SAVE_DIR + 'rewards_std_50s.pkl', "rb") as output_file:
    #     rewards_std = pickle.load(output_file)
    # for i in range(len(rewards_std)):
    #     print(rewards_std[i][149])

    print(saliency_mean)
    print(rewards_mean)

    plot(rewards_mean, rewards_std, saliency_mean, saliency_std)
    plot_correlation(rewards_mean)
