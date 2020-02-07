from baselines.common import atari_wrappers

from toybox.interventions.amidar import *

from saliency_maps.visualize_atari.make_movie import setUp
from saliency_maps.visualize_atari.saliency import score_frame, occlude
from saliency_maps.object_saliency.object_saliency import score_frame_by_pixels
from saliency_maps.jacobian_saliency.jacobian_saliency import get_gradients

from saliency_maps.utils.get_concept_pixels import get_concept_pixels_amidar, world_to_pixels
from saliency_maps.experiments import SAVE_DIR

import numpy as np
import argparse
import pickle
import random
from scipy.misc import imresize

import matplotlib 
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def run_experiment(history, saliency_method='perturbation'):
    print("Setting up trained model")
    env, model = setUp("AmidarToyboxNoFrameskip-v4", "a2c", "./models/AmidarToyboxNoFrameskip-v4/amidar4e7_a2c.model")
    env.reset()
    turtle = atari_wrappers.get_turtle(env)
    tb = turtle.toybox

    saliency_score = {0: {-2:[], -4:[], 0:[], 4:[], 2:[]}, 1: {-2:[], -4:[], 0:[], 4:[], 2:[]}, \
                    2: {-2:[], -4:[], 0:[], 4:[], 2:[]}, 3: {-2:[], -4:[], 0:[], 4:[], 2:[]}, \
                    4: {-2:[], -4:[], 0:[], 4:[], 2:[]}}
    distances = {0: {-2:[], -4:[], 0:[], 4:[], 2:[]}, 1: {-2:[], -4:[], 0:[], 4:[], 2:[]}, \
                    2: {-2:[], -4:[], 0:[], 4:[], 2:[]}, 3: {-2:[], -4:[], 0:[], 4:[], 2:[]}, \
                    4: {-2:[], -4:[], 0:[], 4:[], 2:[]}}

    for i in range(125, len(history['state_json'])):
        state_json = history['state_json'][i]
        frame = history['color_frame'][i]

        #set state to the same state as original game
        tb.write_state_json(state_json)
        enemy_pixels = get_concept_pixels_amidar('enemies', state_json, [frame.shape[1],frame.shape[0]], tb)

        #get saliency
        # S = get_saliency(history, model, i, frame)

        #intervene for each enemy is saliency > 0
        for j,enemy in enumerate(enemy_pixels):
            tb.write_state_json(state_json)
            saliency_orig = get_saliency_on_enemy(history, model, i, frame, enemy, saliency_method=saliency_method)
            
            if saliency_orig > 0:
                dist_orig = get_dist(state_json, tb, j)

                for k in [-8,-6,0,2,4]:
                    tb.write_state_json(state_json)
                    new_state_json, new_color_frame, new_obs = intervention_move_enemy(state_json, env, model, tb, j, move_step=k)

                    if new_state_json is None:
                        continue

                    if k == 0:
                        saliency_score[j][0].append(saliency_orig)
                        distances[j][0].append(dist_orig)  
                        continue

                    plt.imshow(frame)
                    plt.savefig(SAVE_DIR + 'frame{}_e{}'.format(i, j))
                    plt.imshow(new_color_frame)
                    plt.savefig(SAVE_DIR + 'frame{}_e{}_intervene{}'.format(i, j, k))
    
                    saliency = get_saliency_on_enemy(history, model, i, new_color_frame, enemy, inp=new_obs, saliency_method=saliency_method)
                    dist = get_dist(new_state_json, tb, j)

                    if k == -6 or k == -8:
                        saliency_score[j][k+4].append(saliency)
                        distances[j][k+4].append(dist) 
                    else:
                        saliency_score[j][k].append(saliency)
                        distances[j][k].append(dist)

    return saliency_score, distances

def intervention_move_enemy(state_json, env, model, tb, enemy_id, move_step):
    with AmidarIntervention(tb) as intervention: 
        intervention.set_enemy_protocol(enemy_id, 'EnemyAmidarMvmt')
        state_json = intervention.state
        # print('enemy id {}: '.format(enemy_id), state_json['enemies'][enemy_id])
        # print('player {}: ', state_json['player'])

        if state_json['player']['step'] is None or state_json['enemies'][enemy_id]['step'] is None:
            return None, None, None

        player_pos_tx = state_json['player']['step']['tx']
        player_pos_ty = state_json['player']['step']['ty']
        enemy_pos_tx = state_json['enemies'][enemy_id]['step']['tx']
        enemy_pos_ty = state_json['enemies'][enemy_id]['step']['ty']

        #verify whether player and enemy are on same horizontal segment
        if player_pos_ty != enemy_pos_ty:
            return None, None, None

        print("Intervening on enemy {}'s' position now -- moving {} steps".format(enemy_id, move_step))
        print("old next step: ", state_json['enemies'][enemy_id]['step'])

        if move_step < 0: #i.e move closer to player
            if player_pos_tx - enemy_pos_tx > 0: #ie player is on right
                move_step *= -1
        else: #i.e. move farther from player
            if player_pos_tx - enemy_pos_tx > 0: #ie player is on right
                move_step *= -1

        #intervene by moving enemy
        next_step_tx = enemy_pos_tx + move_step
        state_json['enemies'][enemy_id]['step']['tx'] = next_step_tx

        print("new next step: ", state_json['enemies'][enemy_id]['step'])
        tb.write_state_json(state_json)

    #forward simulate 3 steps with no-op action
    for x in range(3):
        obs, _, _, _ = env.step(0)
    # for i in range(4):
    #     plt.imshow(obs[0,:,:,i])
    #     plt.show()
    
    #get logits and reset tb to state_json
    actions, value, _, _, a_logits, _ = model.step(obs)
    new_obs = obs
    # tb.write_state_json(state_json) 
    new_color_frame = tb.get_rgb_frame()
    new_state_json = tb.state_to_json()

    return new_state_json, new_color_frame, new_obs

def get_saliency_on_enemy(history, model, ix, frame, enemy_pixels, inp=None, saliency_method='perturbation'):
    if saliency_method == 'perturbation':
        actor_saliency = score_frame(model, history, ix, r=2, d=5, interp_func=occlude, mode='actor', inp=inp)
        S = np.zeros((110, 84))
        S[18:102, :] = actor_saliency
        S = imresize(actor_saliency, size=[frame.shape[0],frame.shape[1]], interp='bilinear').astype(np.float32)

        salient_pixels = []
        for p in enemy_pixels:
            salient_pixels.append(S[p[1]][p[0]])
        saliency = np.mean(salient_pixels)
    elif saliency_method == 'object':
        saliency = score_frame_by_pixels(model, history, ix, enemy_pixels, inp=inp, mode='actor')
    elif saliency_method == 'jacobian':
        saliency = get_gradients(model, history['ins'][ix], mode='actor')
        S = np.zeros((110, 84))
        S[18:102, :] = saliency[0,:,:,3]**2
        S = imresize(saliency[0,:,:,3]**2, size=[frame.shape[0],frame.shape[1]], interp='bilinear').astype(np.float32)

        salient_pixels = []
        for p in enemy_pixels:
            salient_pixels.append(S[p[1]][p[0]])
        saliency = np.mean(salient_pixels)
        
    return saliency

# def get_saliency_on_enemy(S, enemy_pixels, saliency_method='perturbation'):
#     if saliency_method == 'perturbation':
#         salient_pixels = []
#         for p in enemy_pixels:
#             salient_pixels.append(S[p[1]][p[0]])
#         saliency = np.mean(salient_pixels)
#     else:
        

#     return saliency

def get_dist(state_json, tb, enemy_id):
    enemy = state_json['enemies'][enemy_id]
    enemy_index = (enemy['position']['x'], enemy['position']['y'])
    enemy_pos = world_to_pixels(enemy_index, tb)    

    player_index = (state_json['player']['position']['x'], state_json['player']['position']['y'])
    player_pos = world_to_pixels(player_index, tb)

    distance = abs(player_pos[0] - enemy_pos[0]) + abs(player_pos[1] - enemy_pos[1])

    return distance

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-s', '--saliency_method', default='perturbation', type=str, help='saliency method to be used')
    # parser.add_argument('-n', '--num_frames', default=150, type=int, help='number of frames to be processed')
    # parser.add_argument('-r', '--range', default=[1,51], type=list, help='range of pkl file numbering')
    # parser.add_argument('-a', '--aggregation', default='mean', type=str, help='aggregation type of different samples')
    args = parser.parse_args()

    load_dir = "./saliency_maps/movies/a2c/AmidarToyboxNoFrameskip-v4/perturbation/"
    # history_path = load_dir + "default-250-amidartoyboxnoframeskip-v4-7.pkl"
    history_path = load_dir + "default-1000-amidartoyboxnoframeskip-v4-6.pkl"
    SAVE_DIR = SAVE_DIR + "amidar_enemy_dist_ex/{}/".format(args.saliency_method) 

    with open(history_path, "rb") as output_file:
        history_file = pickle.load(output_file)

    saliency_score, distances = run_experiment(history_file, args.saliency_method)
    filehandler1 = open(SAVE_DIR + 'intervention_saliency.pkl', 'wb') 
    pickle.dump(saliency_score, filehandler1)
    filehandler2 = open(SAVE_DIR + 'intervention_distances.pkl', 'wb') 
    pickle.dump(distances, filehandler2)


