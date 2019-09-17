from saliency_maps.visualize_atari.make_movie import setUp
from saliency_maps.visualize_atari.saliency import score_frame, saliency_on_atari_frame, occlude
from saliency_maps.object_saliency.object_saliency import score_frame_by_pixels
from saliency_maps.jacobian_saliency.jacobian_saliency import get_gradients
from saliency_maps.utils.get_concept_pixels import get_concept_pixels_breakout
from saliency_maps.experiments import CONCEPTS

from baselines.common import atari_wrappers

from toybox.interventions.breakout import *
from toybox.toybox import Toybox, Input

from scipy.misc import imresize # preserves single-pixel info _unlike_ img = img[::2,::2]

import numpy as np
import argparse
import pickle
import random

import matplotlib 
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

#read history and intervene on each timestep
def compute_importance(env_name, alg, model_path, history_path, num_samples, density=5, radius=2, saliency_method='perturbation'):
    #setup model, env, history and dictionary to store data in
    env, model = setUp(env_name, alg, model_path)
    env.reset()
    with open(history_path, "rb") as output_file:
        history = pickle.load(output_file)
    save_dir = "./saliency_maps/experiments/results/CF_imp/"
    if saliency_method=='perturbation':
        save_dir += 'perturbation/'
    elif saliency_method=='object':
        save_dir += 'object/'
    elif saliency_method=='jacobian':
        save_dir += 'jacobian/'
    history_name = history_path.split("/")[-1][:-4]
    print(history_name)

    print("Running through history")
    turtle = atari_wrappers.get_turtle(env)
    tb = turtle.toybox
    concepts = get_env_concepts()

    #run through history
    episode_actionImp = []
    episode_valueImp = []
    for i in range(30, len(history['ins'])):
        concept_actionImp = {}
        concept_valueImp = {}
        tb.write_state_json(history['state_json'][i-1])

        #get raw saliency score
        frame = history['color_frame'][i-1]
        if saliency_method == 'perturbation':
            actor_saliency = score_frame(model, history, i-1, radius, density, interp_func=occlude, mode='actor')
            S_action = np.zeros((110, 84))
            S_action[18:102, :] = actor_saliency
            S_action = imresize(actor_saliency, size=[frame.shape[0],frame.shape[1]], interp='bilinear').astype(np.float32)

            critic_saliency = score_frame(model, history, i-1, radius, density, interp_func=occlude, mode='critic')
            S_value = np.zeros((110, 84))
            S_value[18:102, :] = critic_saliency
            S_value = imresize(critic_saliency, size=[frame.shape[0],frame.shape[1]], interp='bilinear').astype(np.float32)

            #get frame with saliency
            frame = saliency_on_atari_frame(actor_saliency, frame, fudge_factor=300, channel=2) #blue
            frame = saliency_on_atari_frame(critic_saliency, frame, fudge_factor=600, channel=0) #red
            plt.figure()
            plt.imshow(frame)
            plt.savefig(save_dir + history_name + '/num_samples_{}/frame{}.png'.format(num_samples, i))
        elif saliency_method == 'jacobian':
            jacobian_actor = get_gradients(model, history['ins'][i-1], mode='actor')
            S_action = np.zeros((110, 84))
            S_action[18:102, :] = jacobian_actor[0,:,:,3]**2
            S_action = imresize(jacobian_actor[0,:,:,3]**2, size=[frame.shape[0],frame.shape[1]], interp='bilinear').astype(np.float32)

            jacobian_critic = get_gradients(model, history['ins'][i-1], mode='critic')
            S_value = np.zeros((110, 84))
            S_value[18:102, :] = jacobian_critic[0,:,:,3]**2
            S_value = imresize(jacobian_critic[0,:,:,3]**2, size=[frame.shape[0],frame.shape[1]], interp='bilinear').astype(np.float32)

            frame = saliency_on_atari_frame((jacobian_actor[0,:,:,3]**2).squeeze(), frame, fudge_factor=300, channel=2) #blue
            frame = saliency_on_atari_frame((jacobian_critic[0,:,:,3]**2).squeeze(), frame, fudge_factor=50, channel=0) #blue
            plt.figure()
            plt.imshow(frame)
            plt.savefig(save_dir + history_name + '/num_samples_{}/frame{}.png'.format(num_samples, i))

        #go through all objects and get object saliency and CF importance
        frame = history['color_frame'][i-1]
        for concept in concepts:
            concept_actionImp[concept] = {}
            concept_valueImp[concept] = {}

            #get concept location pixels
            concept_pixels = get_concept_pixels_breakout(concept, history['state_json'][i-1], [frame.shape[1],frame.shape[0]])
            
            #change pixels to white to see mapping in the real frame
            # for pixel in concept_pixels:
            #     frame[pixel[1], pixel[0], 0] = 255
            #     frame[pixel[1], pixel[0], 1] = 255
            #     frame[pixel[1], pixel[0], 2] = 255
            # plt.imshow(frame)
            # plt.show()

            score_pixels_actions = []
            score_pixels_values = []
            if saliency_method == 'perturbation' or saliency_method == 'jacobian':
                for pixels in concept_pixels:
                    score_pixels_actions.append(S_action[pixels[1]][pixels[0]])
                    score_pixels_values.append(S_value[pixels[1]][pixels[0]])
                concept_actionImp[concept]["SM_imp"] = np.mean(score_pixels_actions)
                concept_valueImp[concept]["SM_imp"] = np.mean(score_pixels_values)
            elif saliency_method == 'object':
                actor_score = score_frame_by_pixels(model, history, i-1, concept_pixels, mode='actor')
                critic_score = score_frame_by_pixels(model, history, i-1, concept_pixels, mode='critic')
                concept_actionImp[concept]["SM_imp"] = actor_score
                concept_valueImp[concept]["SM_imp"] = critic_score

            #apply interventions to concept
            CF_imp_action, CF_imp_value, CF_IV_intensity = apply_interventions(concept, history['a_logits'][i], history['values'][i], tb, \
                                                history['state_json'][i-1], env, model, concept_pixels, num_samples=num_samples)
            concept_actionImp[concept]["CF_imp"] = CF_imp_action
            concept_valueImp[concept]["CF_imp"] = CF_imp_value
            concept_actionImp[concept]["IV_intensity"] = CF_IV_intensity
            concept_valueImp[concept]["IV_intensity"] = CF_IV_intensity

        episode_actionImp += [concept_actionImp]
        episode_valueImp += [concept_valueImp]

    #save data
    print(episode_actionImp)
    print(episode_valueImp)
    filehandler1 = open(save_dir + history_name + '/num_samples_{}/episode_actionImp.pkl'.format(num_samples), 'wb') 
    pickle.dump(episode_actionImp, filehandler1)
    filehandler2 = open(save_dir + history_name + '/num_samples_{}/episode_valueImp.pkl'.format(num_samples), 'wb') 
    pickle.dump(episode_valueImp, filehandler2)

def get_env_concepts():
    return CONCEPTS["Breakout"]

def apply_interventions(concept, a_logits, values, tb, state_json, env, model, pixels, num_samples):
    global INTERVENTIONS

    CF_imp_action = []
    CF_imp_value = []
    CF_IV_intensity = []
    if "bricks" in concept:
        concept = "bricks"
    interventions = INTERVENTIONS[concept]

    for i in range(num_samples):
        IV_a_logits = []
        IV_values = []
        CF_intensity = []
        CF_actions = []
        CF_values = []
        #get a_logits from interventions
        for IV in interventions:
            logits, value, intensity = IV(tb, state_json, env, model, pixels)
            IV_a_logits += [logits]
            IV_values += [value]
            CF_intensity += [intensity]

        #get euclidean distance of a_logits and value before and after intervention
        for j,IV_logits in enumerate(IV_a_logits):
            euc_dist_action = np.linalg.norm(IV_logits - a_logits)
            euc_dist_value = np.linalg.norm(IV_values[j] - values)
            CF_actions += [euc_dist_action]
            CF_values += [euc_dist_value]
        print("CF_actions: ", CF_actions)
        print("CF_values: ", CF_values)
        print("CF_intensity: ", CF_intensity)
        CF_imp_action += [CF_actions]
        CF_imp_value += [CF_values]
        CF_IV_intensity += [CF_intensity]

    return CF_imp_action, CF_imp_value, CF_IV_intensity

def intervention_move_ball(tb, state_json, env, model, pixels):
    distances = range(5,16)

    print("Intervening on ball position now and forward simulating")
    with BreakoutIntervention(tb) as intervention: 
        move_distance = random.choice(distances)
        ball_pos = intervention.get_ball_position()
        # print("old: ", ball_pos)
        ball_pos['x'] = ball_pos['x'] + move_distance
        ball_pos['y'] = ball_pos['y'] + move_distance
        # print("new: ", ball_pos)
        intervention.set_ball_position(ball_pos)
        ball_pos_post = intervention.get_ball_position()
        assert ball_pos_post['x'] == ball_pos['x']

    #forward simulate 3 steps with no-op action
    for i in range(3):
        obs, _, _, _ = env.step(0)

    #get logits and reset tb to state_json
    actions, value, _, _, a_logits, _ = model.step(obs)
    tb.write_state_json(state_json) 

    return list(a_logits), value, move_distance

def intervention_ball_speed(tb, state_json, env, model, pixels):
    delta = range(1,4)

    print("Intervening on ball velocity now and forward simulating")
    with BreakoutIntervention(tb) as intervention: 
        delta_velocity = random.choice(delta)
        ball_vel = intervention.get_ball_velocity()
        ball_vel['x'] = ball_vel['x'] + delta_velocity
        ball_vel['y'] = ball_vel['y'] + delta_velocity
        intervention.set_ball_velocity(ball_vel)
        ball_vel_post = intervention.get_ball_velocity()
        assert ball_vel_post['x'] == ball_vel['x']

    #forward simulate 3 steps with no-op action
    for i in range(3):
        obs, _, _, _ = env.step(0)

    #get logits and reset tb to state_json
    actions, value, _, _, a_logits, _ = model.step(obs)
    tb.write_state_json(state_json) 

    return list(a_logits), value, delta_velocity

def intervention_move_paddle(tb, state_json, env, model, pixels):
    distances = range(5,16)

    print("Intervening on paddle position now and forward simulating")
    with BreakoutIntervention(tb) as intervention: 
        move_distance = random.choice(distances)
        direction = random.choice(range(2))

        pos = intervention.get_paddle_position()
        # print("old: ", pos)
        if direction != 0:
            move_distance *= -1
        pos['x'] = pos['x'] + move_distance
        # print("new: ", pos)
        intervention.set_paddle_position(pos)

    #forward simulate 3 steps with no-op action
    for i in range(3):
        obs, _, _, _ = env.step(0)

    #get logits and reset tb to state_json
    actions, value, _, _, a_logits, _ = model.step(obs)
    tb.write_state_json(state_json) 

    return list(a_logits), value, move_distance

def intervention_flip_bricks(tb, state_json, env, model, pixels):
    bricks_in_pixels, brick_indices = get_pixel_bricks(tb, pixels)
    num_dead_bricks = 0

    print("Intervening on flipping bricks now and forward simulating")
    with BreakoutIntervention(tb) as intervention: 
        for i in range(len(bricks_in_pixels)):
            if bricks_in_pixels[i]['alive']:
                intervention.set_brick(brick_indices[i], alive=False)
            else:
                num_dead_bricks += 1
                intervention.set_brick(brick_indices[i], alive=True)

    #forward simulate 3 steps with no-op action
    for i in range(3):
        obs, _, _, _ = env.step(0)

    #get logits and reset tb to state_json
    actions, value, _, _, a_logits, _ = model.step(obs)
    tb.write_state_json(state_json) 

    return list(a_logits), value, num_dead_bricks

def intervention_remove_bricks(tb, state_json, env, model, pixels):
    #select 5 random bricks from 18 bricks in sub space
    remove_bricks = random.sample(range(18), 5)
    bricks_in_pixels, brick_indices = get_pixel_bricks(tb, pixels)
    remove_bricks_depth = 0

    print("Intervening on removing bricks now and forward simulating")
    with BreakoutIntervention(tb) as intervention: 
        for i in remove_bricks:
            if bricks_in_pixels[i]['alive']:
                # print("Removing brick row: {}, col: {}".format(bricks_in_pixels[i]['row'],bricks_in_pixels[i]['col']))
                remove_bricks_depth += bricks_in_pixels[i]['points']
                intervention.set_brick(brick_indices[i], alive=False)

    #forward simulate 3 steps with no-op action
    for i in range(3):
        obs, _, _, _ = env.step(0)

    #get logits and reset tb to state_json
    actions, value, _, _, a_logits, _ = model.step(obs)
    tb.write_state_json(state_json) 

    return list(a_logits), value, remove_bricks_depth

def intervention_remove_rows(tb, state_json, env, model, pixels):
    bricks_in_pixels, brick_indices = get_pixel_bricks(tb, pixels)

    print("Intervening on removing rows of bricks now and forward simulating")
    with BreakoutIntervention(tb) as intervention: 
        #select a row to remove from subspace
        remove_row = random.choice(range(3))
        for i in range(len(bricks_in_pixels)):
            if bricks_in_pixels[i]['row'] == remove_row or bricks_in_pixels[i]['row'] == remove_row + 3:
                intervention.set_brick(brick_indices[i], alive=False)

    #forward simulate 3 steps with no-op action
    for i in range(3):
        obs, _, _, _ = env.step(0)

    #get logits and reset tb to state_json
    actions, value, _, _, a_logits, _ = model.step(obs)
    tb.write_state_json(state_json) 

    return list(a_logits), value, remove_row

def intervention_add_channel(tb, state_json, env, model, pixels):
    return None

def get_pixel_bricks(tb, pixels):
    bricks_in_pixels = []
    brick_indices = []

    with BreakoutIntervention(tb) as intervention: 
        all_bricks = intervention.get_bricks()
        for i, brick in enumerate(all_bricks):
            if (brick['position']['x'] >= pixels[0][0] and \
                brick['position']['y'] >= pixels[0][1]) and \
                (brick['position']['x'] + brick['size']['x'] <= pixels[-1][0] + 1 and \
                 brick['position']['y'] + brick['size']['y'] <= pixels[-1][1] + 1):
                bricks_in_pixels += [brick]
                brick_indices += [i]

        assert len(bricks_in_pixels) == 18
        assert len(brick_indices) == 18

    return bricks_in_pixels, brick_indices

#BAD CODE PRACTICE!!
INTERVENTIONS = {"balls": [intervention_move_ball, intervention_ball_speed], \
                "paddle": [intervention_move_paddle], \
                "bricks": [intervention_flip_bricks, intervention_remove_bricks, intervention_remove_rows]}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-e', '--env_name', default='BreakoutToyboxNoFrameskip-v4', type=str, help='name of gym environment')
    parser.add_argument('-n', '--num_samples', default=10, type=int, help='number of samples to compute importance over')
    parser.add_argument('-s', '--saliency_method', default="perturbation", help='saliency method to be used')
    parser.add_argument('-a', '--alg', help='algorithm used for training')
    parser.add_argument('-l', '--load_path', help='path to load the model from')
    parser.add_argument('-hp', '--history_path', help='path of history of a executed episode')
    args = parser.parse_args()

    compute_importance(args.env_name, args.alg, args.load_path, args.history_path, args.num_samples, saliency_method=args.saliency_method)
