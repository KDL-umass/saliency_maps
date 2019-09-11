from baselines.common import atari_wrappers

from toybox.interventions.amidar import *
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
        print(history['state_json'][i])
        for concept in concepts:
            interventions_imp = []

            #get concept location pixels
            concept_pixels = get_concept_pixels(concept, history['state_json'][i], [frame.shape[1],frame.shape[0]], tb)
            print(concept_pixels)
            
            #change pixels to white to see mapping in the real frame
            for pixel in concept_pixels:
                frame[pixel[1], pixel[0], 0] = 255
                frame[pixel[1], pixel[0], 1] = 255
                frame[pixel[1], pixel[0], 2] = 255

            #get raw saliency score
            # frame = history['color_frame'][i]
            # actor_saliency = score_frame(model, history, i, radius, density, interp_func=occlude, mode='actor')
            # S = np.zeros((110, 84))
            # S[18:102, :] = actor_saliency
            # S = imresize(actor_saliency, size=[frame.shape[0],frame.shape[1]], interp='bilinear').astype(np.float32)
            # # print(frame.shape, S.shape)
            # score_pixels = []
            # for pixels in concept_pixels:
            #     score_pixels.append(S[pixels[1]][pixels[0]])
            # print(score_pixels)
            # SM_imp.append(np.mean(score_pixels))

            # #apply intervention to concept
            # apply_intervention(history['state_json'][i], concept, a_logits, tb)
        plt.imshow(frame)
        plt.show()  

def get_env_concepts():
    return CONCEPTS["Amidar"]

def get_concept_pixels(concept, state_json, size, tb):
    pixels = []
    board_width = size[0]
    board_length = size[1]
    print("concept: ", concept)
    ["tiles", "player", "enemies", "score", "lives"]

    if concept == "tiles":
        return []
    elif concept == "player":
        player_index = (state_json[concept]['position']['x'], state_json[concept]['position']['y']) #world pos
        player_pos = world_to_pixels(player_index, tb) #get pixel pos
        #get pixels of tile the player is sitting on
        for x in range(5):
            for y in range(6):
                pixels += [(player_pos[0]+x, player_pos[1]+y)]
        #get pixels outside of tile the player is sitting on
        above_tile = (player_pos[0], player_pos[1]-1)
        below_tile = (player_pos[0], player_pos[1]+1)
        left_tile = (player_pos[0]-1, player_pos[1])
        right_tile = (player_pos[0]+1, player_pos[1])
        pixels += [above_tile, below_tile, left_tile, right_tile]
    elif concept == "enemies":
        for enemy in state_json[concept]:
            enemy_index = (enemy['position']['x'], enemy['position']['y']) #world pos
            enemy_pos = world_to_pixels(enemy_index, tb) #get pixel pos
            #get pixels of tile the enemy is sitting on
            for x in range(5):
                for y in range(6):
                    pixels += [(enemy_pos[0]+x, enemy_pos[1]+y)]
            #get pixels outside of tile the enemy is sitting on
            above_tile = (enemy_pos[0], enemy_pos[1]-1)
            below_tile = (enemy_pos[0], enemy_pos[1]+1)
            left_tile = (enemy_pos[0]-1, enemy_pos[1])
            right_tile = (enemy_pos[0]+1, enemy_pos[1])
            pixels += [above_tile, below_tile, left_tile, right_tile]
    elif concept == "score":
        for x in range(80,105):
            for y in range(195,210):
                pixels += [(x,y)]
    elif concept == "lives":
        for x in range(115,150):
            for y in range(195,210):
                pixels += [(x,y)]

    return pixels

def apply_intervention(concept, a_logits, tb, state_json, env, model, num_samples):
    CF_imp_concept = []
    CF_IV_intensity = []
    interventions = INTERVENTIONS[concept]

    for i in range(num_samples):
        IV_a_logits = []
        CF_intensity = []
        CF_imp = []
        #get a_logits from interventions
        for IV in interventions:
            logits, intensity = IV(tb, state_json, env, model)
            IV_a_logits += [logits]
            CF_intensity += [intensity]

        #get euclidean distance of a_logits before and after intervention
        for IV_logits in IV_a_logits:
            euc_dist = np.linalg.norm(IV_logits - a_logits)
            CF_imp += [euc_dist]
        print("CF_imp: ", CF_imp)
        print("CF_intensity: ", CF_intensity)
        CF_imp_concept += [CF_imp]
        CF_IV_intensity += [CF_intensity]

    return CF_imp_concept, CF_IV_intensity

def intervention_modify_scores(tb, state_json, env, model, pixels):
    delta = range()
    print("Intervening on score now and forward simulating")
    print("old: ", tb.state_to_json()['score'])
    if random_score:
        state_json['score'] = random.randint(1,201)
    else:   
        state_json['score'] = abs_score
    tb.write_state_json(state_json)
    print("new: ", tb.state_to_json()['score'])

    #forward simulate 3 steps with no-op action
    for i in range(3):
        obs, _, _, _ = env.step(0)

    #get logits and reset tb to state_json
    actions, value, _, _, a_logits, _ = model.step(obs)
    tb.write_state_json(state_json) 

    return list(a_logits), move_distance

def intervention_modify_scores_rand(tb, state_json, env, model, pixels):
    delta = range()
    print("Intervening on score now and forward simulating")
    print("old: ", tb.state_to_json()['score'])
    if random_score:
        state_json['score'] = random.randint(1,201)
    else:   
        state_json['score'] = abs_score
    tb.write_state_json(state_json)
    print("new: ", tb.state_to_json()['score'])

    #forward simulate 3 steps with no-op action
    for i in range(3):
        obs, _, _, _ = env.step(0)

    #get logits and reset tb to state_json
    actions, value, _, _, a_logits, _ = model.step(obs)
    tb.write_state_json(state_json) 

    return list(a_logits), move_distance

def intervention_nonchanging_scores(tb, state_json, env, model, pixels):
    delta = range()
    print("Intervening on score now and forward simulating")
    print("old: ", tb.state_to_json()['score'])
    if random_score:
        state_json['score'] = random.randint(1,201)
    else:   
        state_json['score'] = abs_score
    tb.write_state_json(state_json)
    print("new: ", tb.state_to_json()['score'])

    #forward simulate 3 steps with no-op action
    for i in range(3):
        obs, _, _, _ = env.step(0)

    #get logits and reset tb to state_json
    actions, value, _, _, a_logits, _ = model.step(obs)
    tb.write_state_json(state_json) 

    return list(a_logits), move_distance

def intervention_decrement_score(tb, state_json, env, model, pixels):
    decrement_value = range(1,10)

    print("Intervening on score decrement now and forward simulating")
    print("old: ", tb.state_to_json()['score'])
    if random_score:
        state_json['score'] = random.randint(1,201)
    else:   
        state_json['score'] = abs_score
    tb.write_state_json(state_json)
    print("new: ", tb.state_to_json()['score'])

    #forward simulate 3 steps with no-op action
    for i in range(3):
        obs, _, _, _ = env.step(0)

    #get logits and reset tb to state_json
    actions, value, _, _, a_logits, _ = model.step(obs)
    tb.write_state_json(state_json) 

    return list(a_logits), move_distance

def world_to_pixels(world_pos, tb):
    tile_pos = (0, 0)
    with AmidarIntervention(tb) as intervention:
        tile_pos = intervention.world_to_tile(world_pos[0], world_pos[1])
    pixel_pos = (tile_pos['tx']*4 + 16, tile_pos['ty']*5 + 37)

    return pixel_pos

#BAD CODE PRACTICE!!
INTERVENTIONS = {"score": [intervention_modify_scores, intervention_modify_scores_rand, intervention_nonchanging_scores, intervention_decrement_score]}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-e', '--env_name', default='AmidarToyboxNoFrameskip-v4', type=str, help='name of gym environment')
    parser.add_argument('-a', '--alg', help='algorithm used for training')
    parser.add_argument('-l', '--load_path', help='path to load the model from')
    parser.add_argument('-hp', '--history_path', help='path of history of a executed episode')
    args = parser.parse_args()

    compute_importance(args.env_name, args.alg, args.load_path, args.history_path)
