from baselines.common import atari_wrappers

from toybox.interventions.breakout import *
from toybox.toybox import Toybox, Input

from saliency_maps.visualize_atari.make_movie import *
from saliency_maps.visualize_atari.saliency import *
from saliency_maps.experiments import CONCEPTS

from scipy.misc import imresize # preserves single-pixel info _unlike_ img = img[::2,::2]

import numpy as np
import argparse
import pickle
import random

import matplotlib 
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

#read history and intervene on each timestep
def compute_importance(env_name, alg, model_path, history_path, density=5, radius=2):
    #setup model, env and history
    env, model = setUp(env_name, alg, model_path)
    env.reset()
    with open(history_path, "rb") as output_file:
        history = pickle.load(output_file)

    print("Running through history")
    turtle = atari_wrappers.get_turtle(env)
    tb = turtle.toybox
    concepts = get_env_concepts()

    #run through history
    print("number of frames: ", len(history['ins']))
    for i in range(30, len(history['ins'])):
        SM_imp = []
        CF_imp = []
        tb.write_state_json(history['state_json'][i])

        #go through all objects
        frame = history['color_frame'][i]
        for concept in concepts:
            interventions_imp = []
            print(history['state_json'][i])

            #get concept location pixels
            concept_pixels = get_concept_pixels(concept, history['state_json'][i])
            print(concept_pixels)
            
            #change pixels to white to see mapping in the real frame
            for pixel in concept_pixels:
                frame[pixel[1], pixel[0], 0] = 255
                frame[pixel[1], pixel[0], 1] = 255
                frame[pixel[1], pixel[0], 2] = 255

            #get raw saliency score
            frame = history['color_frame'][i]
            actor_saliency = score_frame(model, history, i, radius, density, interp_func=occlude, mode='actor')
            S = np.zeros((110, 84))
            S[18:102, :] = actor_saliency
            S = imresize(actor_saliency, size=[frame.shape[0],frame.shape[1]], interp='bilinear').astype(np.float32)
            # print(frame.shape, S.shape)
            score_pixels = []
            for pixels in concept_pixels:
                score_pixels.append(S[pixels[1]][pixels[0]])
            print(score_pixels)
            SM_imp.append(np.mean(score_pixels))

            #apply intervention to concept
            apply_intervention(history['state_json'][i], concept, a_logits, tb)

        plt.imshow(frame)
        plt.show()

def get_env_concepts():
    return CONCEPTS["Amidar"]

def apply_intervention(state_json, concept, a_logits, tb):
    # if concept=="BreakoutToyboxNoFrameskip-v4":
    return None

def get_concept_pixels(concept, state_json):
    pixels = []
    print("concept: ", concept)

    return pixels

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-e', '--env_name', default='BreakoutToyboxNoFrameskip-v4', type=str, help='name of gym environment')
    parser.add_argument('-a', '--alg', help='algorithm used for training')
    parser.add_argument('-l', '--load_path', help='path to load the model from')
    parser.add_argument('-hp', '--history_path', help='path of history of a executed episode')
    args = parser.parse_args()

    compute_importance(args.env_name, args.alg, args.load_path, args.history_path)
