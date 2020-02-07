from saliency_maps.experiments import SAVE_DIR
from saliency_maps.visualize_atari.saliency import score_frame, saliency_on_atari_frame, occlude
from saliency_maps.jacobian_saliency.jacobian_saliency import get_gradients

from saliency_maps.experiments.CFimportance_breakout import setUp

import numpy as np
import argparse
import pickle
import random

import matplotlib 
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-s', '--saliency_method', default='perturbation', type=str, help='saliency method to be used')
    args = parser.parse_args()

    default_history = './saliency_maps/movies/a2c/BreakoutToyboxNoFrameskip-v4/perturbation/default-150-breakouttoyboxnoframeskip-v4-5.pkl'
    load_dir = "./saliency_maps/movies/a2c/BreakoutToyboxNoFrameskip-v4/perturbation/IVshiftBricks/"
    history_path = load_dir + "IVshiftBricks-150-breakouttoyboxnoframeskip-v4-5_s{}.pkl"
    SAVE_DIR = SAVE_DIR + "breakout_shiftBrick_ex/{}/".format(args.saliency_method) 

    env, model = setUp("BreakoutToyboxNoFrameskip-v4", "a2c", "./models/BreakoutToyboxNoFrameskip-v4/breakout4e7_a2c.model")
    
    for i in range(18):
        if i == 0:
            with open(default_history, "rb") as output_file:
                history = pickle.load(output_file)
        else:
            path = history_path.format(i)
            with open(path, "rb") as output_file:
                history = pickle.load(output_file)

        for j in range(120, 126):
            frame = history['color_frame'][j]
            if args.saliency_method == 'perturbation':
                actor_saliency = score_frame(model, history, j, r=2, d=5, interp_func=occlude, mode='actor')
                critic_saliency = score_frame(model, history, j, r=2, d=5, interp_func=occlude, mode='critic')

                frame = saliency_on_atari_frame(actor_saliency, frame, fudge_factor=300, channel=2)
                frame = saliency_on_atari_frame(critic_saliency, frame, fudge_factor=600, channel=0)

                plt.imshow(frame)
                plt.savefig(SAVE_DIR + 'frame{}_s{}'.format(j, i))
            elif args.saliency_method == 'object':
                pass
            elif args.saliency_method == 'jacobian':
                jacobian_actor = get_gradients(model, history['ins'][j], mode='actor')
                jacobian_critic = get_gradients(model, history['ins'][j], mode='critic')

                frame = saliency_on_atari_frame((jacobian_actor[0,:,:,3]**2).squeeze(), frame, fudge_factor=300, channel=2) #blue
                frame = saliency_on_atari_frame((jacobian_critic[0,:,:,3]**2).squeeze(), frame, fudge_factor=50, channel=0) #blue

                plt.imshow(frame)
                plt.savefig(SAVE_DIR + 'frame{}_s{}'.format(j, i))
                
    # history_path = './saliency_maps/movies/a2c/BreakoutToyboxNoFrameskip-v4/perturbation/IVsymbricks-250-breakouttoyboxnoframeskip-v4-3.pkl'
    # SAVE_DIR = SAVE_DIR + "breakout_flipBrick_ex/"
    # with open(history_path, "rb") as output_file:
    #     history = pickle.load(output_file)

    # for j in range(115, 126):
    #     frame = history['color_frame'][j]
    #     actor_saliency = score_frame(model, history, j, r=2, d=5, interp_func=occlude, mode='actor')
    #     critic_saliency = score_frame(model, history, j, r=2, d=5, interp_func=occlude, mode='critic')

    #     frame = saliency_on_atari_frame(actor_saliency, frame, fudge_factor=300, channel=2)
    #     frame = saliency_on_atari_frame(critic_saliency, frame, fudge_factor=600, channel=0)

    #     plt.imshow(frame)
    #     plt.savefig(SAVE_DIR + 'frame{}_bp'.format(j))


