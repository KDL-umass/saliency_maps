from baselines.common import atari_wrappers

from toybox.interventions.breakout import *
from toybox.toybox import Toybox, Input

from saliency_maps.visualize_atari.make_movie import *
from saliency_maps.visualize_atari.saliency import *

import numpy as np
from scipy.misc import imresize # preserves single-pixel info _unlike_ img = img[::2,::2]
import argparse
import pickle

CONCEPTS = {"Breakout": ["paddle", "brick"], "Amidar": ["tiles", "player", "enemies", "score", "lives"]}

#read history and intervene on each timestep
def compute_importance(env_name, alg, model_path, history_path, density=5, radius=2):
    #setup model, env and history
    env, model = setUp(env_name, alg, model_path)
    with open(history_path, "rb") as output_file:
        history = pickle.load(output_file)

    print("Running through history")
    turtle = atari_wrappers.get_turtle(env)
    tb = turtle.toybox
    concepts = get_env_concepts(env_name)

    #run through history
    for i in range(10, len(history['ins'])):
        SM_imp = []
        CF_imp = []
        tb.write_state_json(history['state_json'][i])

        #go through all objects
        for concept in concepts:
            interventions_imp = []
            print(history['state_json'][i])

            #get concept location pixels
            concept_pixels = []
            if concept == "balls":
                ball_pos = (history['state_json'][i][concept][0]['position']['x'], history['state_json'][i][concept][0]['position']['y'])
                ball_radius = history['state_json'][i]['ball_radius']
                for i in range(ball_radius):
                    new_pos1 = (paddle_pos[0] + i, paddle_pos[1])
                    new_pos2 = (paddle_pos[0], paddle_pos[1] + i)
                    new_pos3 = (paddle_pos[0] + i, paddle_pos[1] + i)
                    concept_pixels.append(new_pos1)
                    concept_pixels.append(new_pos2)
                    concept_pixels.append(new_pos3)
            elif concept == "paddle":
                paddle_pos = (int(history['state_json'][i][concept]['position']['x']), int(history['state_json'][i][concept]['position']['y']))
                paddle_width = history['state_json'][i]['paddle_width']
                concept_pixels.append(paddle_pos)
                print(paddle_width)
                for i in range(1, int(paddle_width/2)):
                    left_pos = (int(paddle_pos[0] - i), int(paddle_pos[1]))
                    right_pos = (int(paddle_pos[0] + i), int(paddle_pos[1]))
                    concept_pixels.append(left_pos)
                    concept_pixels.append(right_pos)
            print(concept_pixels)

            #get raw saliency score
            frame = history['color_frame'][i]
            actor_saliency = score_frame(model, history, i, radius, density, interp_func=occlude, mode='actor')
            S = np.zeros((110, 84))
            S[18:102, :] = actor_saliency
            S = imresize(actor_saliency, size=[frame.shape[0],frame.shape[1]], interp='bilinear').astype(np.float32)
            score_pixels = []
            for pixels in concept_pixels:
                score_pixels.append(S[pixels[0]][pixels[1]])
            print(score_pixels)
            SM_imp.append(np.mean(score_pixels))

            #actions, value, _, _, a_logits = model.step(history['ins'][i])

    '''
    while not done and episode_length <= max_ep_len:
        episode_length += 1
        actions, value, _, _, a_logits = model.step(obs)
        num_lives = turtle.ale.lives()
        obs, reward, done, info = env.step(actions)
        epr += reward[0]
        color_frame = turtle.toybox.get_rgb_frame()
        state_json = tb.state_to_json()
    '''

def get_env_concepts(env_name):
    global CONCEPTS

    if env_name=="BreakoutToyboxNoFrameskip-v4":
        return CONCEPTS["Breakout"]
    elif env_name=="AmidarToyboxNoFrameskip-v4":
        return CONCEPTS["Amidar"]
    else:
        print('environment "{}" not supported'.format(env_name))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-e', '--env_name', default='BreakoutToyboxNoFrameskip-v4', type=str, help='name of gym environment')
    parser.add_argument('-a', '--alg', help='algorithm used for training')
    parser.add_argument('-l', '--load_path', help='path to load the model from')
    parser.add_argument('-hp', '--history_path', help='path of history of a executed episode')
    args = parser.parse_args()

    compute_importance(args.env_name, args.alg, args.load_path, args.history_path)
