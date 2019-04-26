from baselines.common import atari_wrappers

from toybox.interventions.breakout import *
from toybox.toybox import Toybox, Input

from saliency_maps.visualize_atari.make_movie import *
from saliency_maps.visualize_atari.saliency import *

import numpy as np
from scipy.misc import imresize # preserves single-pixel info _unlike_ img = img[::2,::2]
import argparse
import pickle
import matplotlib 
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

CONCEPTS = {"Breakout": ["balls", "paddle", "brick"], "Amidar": ["tiles", "player", "enemies", "score", "lives"]}

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
    print("number of frames: ", len(history['ins']))
    for i in range(30, len(history['ins'])):
        SM_imp = []
        CF_imp = []
        tb.write_state_json(history['state_json'][i])

        #go through all objects

        frame = history['color_frame'][i]
        for concept in concepts:
            interventions_imp = []
            # print(history['state_json'][i])

            #get concept location pixels
            concept_pixels = get_concept_pixels(env_name, concept, history['state_json'][i])
            print(concept_pixels)
            
            #change pixels to white to see mapping in the real frame
            for pixel in concept_pixels:
                frame[pixel[1], pixel[0], 0] = 255
                frame[pixel[1], pixel[0], 1] = 255
                frame[pixel[1], pixel[0], 2] = 255
        plt.imshow(frame)
        plt.show()
        '''
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
            SM_imp.append(np.mean(score_pixels))'''

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

def get_concept_pixels(env_name, concept, state_json):
    if env_name=="BreakoutToyboxNoFrameskip-v4":
        return get_breakout_concept_pixels(concept, state_json)
    elif env_name=="AmidarToyboxNoFrameskip-v4":
        return get_amidar_concept_pixels(concept, state_json)
    else:
        print('environment "{}" not supported'.format(env_name))

def get_breakout_concept_pixels(concept, state_json):
    pixels = []
    print("concept: ", concept)
    print(state_json[concept])

    if concept == "balls":
        ball_pos = (int(state_json[concept][0]['position']['x']), int(state_json[concept][0]['position']['y']))
        ball_radius = int(state_json['ball_radius'])
        pixels += [ball_pos]
        print("ball radius: ", ball_radius)

        for i in range(1, ball_radius+1):
            right_pos = (ball_pos[0] + i, ball_pos[1])
            left_pos = (ball_pos[0]  - i, ball_pos[1])
            lower_pos = (ball_pos[0], ball_pos[1] + i)
            upper_pos = (ball_pos[0], ball_pos[1] - i)
            lower_right_pos = (ball_pos[0] + i, ball_pos[1] + i)
            lower_left_pos = (ball_pos[0] - i, ball_pos[1] + i)
            upper_left_pos = (ball_pos[0] - i, ball_pos[1] - i)
            upper_right_pos = (ball_pos[0] + i, ball_pos[1] - i)

            pixels += [right_pos, left_pos , lower_pos, upper_pos, lower_right_pos, lower_left_pos, upper_left_pos, upper_right_pos]
    elif concept == "paddle":
        paddle_pos = (int(state_json[concept]['position']['x']), int(state_json[concept]['position']['y']))
        paddle_width = int(state_json['paddle_width'])
        #pixels.append(paddle_pos)
        print("paddle width: ", paddle_width)

        for i in range(int(paddle_width/2)+1):
            left_pos = (paddle_pos[0] - i, paddle_pos[1])
            right_pos = (paddle_pos[0] + i, paddle_pos[1])
            uleft_pos = (paddle_pos[0] - i, paddle_pos[1] + 1)
            uright_pos = (paddle_pos[0] + i, paddle_pos[1] + 1)
            uuleft_pos = (paddle_pos[0] - i, paddle_pos[1] + 2)
            uuright_pos = (paddle_pos[0] + i, paddle_pos[1] + 2)
            if i == 0:
                pixels += [left_pos, uleft_pos, uright_pos, uuleft_pos, uuright_pos]
            else:
                pixels += [left_pos, right_pos, uleft_pos, uright_pos, uuleft_pos, uuright_pos]

    return pixels

def get_amidar_concept_pixels(concept, state_json):
    pixels = []
    return pixels

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-e', '--env_name', default='BreakoutToyboxNoFrameskip-v4', type=str, help='name of gym environment')
    parser.add_argument('-a', '--alg', help='algorithm used for training')
    parser.add_argument('-l', '--load_path', help='path to load the model from')
    parser.add_argument('-hp', '--history_path', help='path of history of a executed episode')
    args = parser.parse_args()

    compute_importance(args.env_name, args.alg, args.load_path, args.history_path)
