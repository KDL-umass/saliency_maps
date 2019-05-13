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
def compute_importance(env_name, alg, model_path, history_path, num_samples, density=5, radius=2):
    #setup model, env, history and dictionary to store data in
    env, model = setUp(env_name, alg, model_path)
    env.reset()
    with open(history_path, "rb") as output_file:
        history = pickle.load(output_file)
    save_dir = "./saliency_maps/experiments/results/"
    history_name = history_path.split("/")[-1][:-4]
    print(history_name)

    print("Running through history")
    turtle = atari_wrappers.get_turtle(env)
    tb = turtle.toybox
    concepts = get_env_concepts()

    #run through history
    episode_importance = []
    for i in range(30, len(history['ins'])):
        concept_imp = {}
        tb.write_state_json(history['state_json'][i])

        #go through all objects
        frame = history['color_frame'][i]
        for concept in concepts:
            concept_imp[concept] = {}

            #get concept location pixels
            concept_pixels = get_concept_pixels(concept, history['state_json'][i], [frame.shape[1],frame.shape[0]])
            
            #change pixels to white to see mapping in the real frame
            # for pixel in concept_pixels:
            #     frame[pixel[1], pixel[0], 0] = 255
            #     frame[pixel[1], pixel[0], 1] = 255
            #     frame[pixel[1], pixel[0], 2] = 255
            # plt.imshow(frame)
            # plt.show()

            #get raw saliency score
            frame = history['color_frame'][i]
            actor_saliency = score_frame(model, history, i, radius, density, interp_func=occlude, mode='actor')
            S = np.zeros((110, 84))
            S[18:102, :] = actor_saliency
            S = imresize(actor_saliency, size=[frame.shape[0],frame.shape[1]], interp='bilinear').astype(np.float32)
            score_pixels = []
            for pixels in concept_pixels:
                score_pixels.append(S[pixels[1]][pixels[0]])
            concept_imp[concept]["SM_imp"] = np.mean(score_pixels)

            #get frame with saliency
            frame = saliency_on_atari_frame(actor_saliency, frame, fudge_factor=300, channel=2) #blue
            plt.figure()
            plt.imshow(frame)
            plt.savefig(save_dir + history_name + '/num_samples_{}/frame{}.png'.format(num_samples, i))

            #apply interventions to concept
            CF_imp_concept, CF_IV_intensity = apply_interventions(concept, history['a_logits'][i], tb, history['state_json'][i], \
                                                env, model, concept_pixels, num_samples=num_samples)
            concept_imp[concept]["CF_imp"] = CF_imp_concept
            concept_imp[concept]["IV_intensity"] = CF_IV_intensity

        episode_importance += [concept_imp]

    #save data
    print(episode_importance)
    filehandler = open(save_dir + history_name + '/num_samples_{}/episode_importance.pkl'.format(num_samples), 'wb') 
    pickle.dump(episode_importance, filehandler)

def get_env_concepts():
    return CONCEPTS["Breakout"]

def get_concept_pixels(concept, state_json, size):
    pixels = []

    if concept == "balls":
        ball_pos = (int(state_json[concept][0]['position']['x']), int(state_json[concept][0]['position']['y']))
        ball_radius = int(state_json['ball_radius'])
        pixels += [ball_pos]

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

        for i in range(int(paddle_width/2)+1):
            left_pos = (paddle_pos[0] - i, paddle_pos[1])
            right_pos = (paddle_pos[0] + i, paddle_pos[1])
            lleft_pos = (paddle_pos[0] - i, paddle_pos[1] + 1)
            lright_pos = (paddle_pos[0] + i, paddle_pos[1] + 1)
            llleft_pos = (paddle_pos[0] - i, paddle_pos[1] + 2)
            llright_pos = (paddle_pos[0] + i, paddle_pos[1] + 2)
            if i == 0:
                pixels += [left_pos, lleft_pos, lright_pos, llleft_pos, llright_pos]
            else:
                pixels += [left_pos, right_pos, lleft_pos, lright_pos, llleft_pos, llright_pos]
    elif concept == "bricks_top_left":
        bricks = state_json["bricks"]
        brick_size = (int(bricks[0]['size']['x']), int(bricks[0]['size']['y'])) 
        upper_left_corner = (int(bricks[0]['position']['x']), int(bricks[0]['position']['y']))
        lower_right_corner = (int(bricks[-1]['position']['x']) + brick_size[0] - 1, int(bricks[-1]['position']['y']) + brick_size[1] - 1)

        x_partition = int((lower_right_corner[0] - upper_left_corner[0] + 1)/3)
        y_partition = int((lower_right_corner[1] - upper_left_corner[1] + 1)/2)

        for x in range(x_partition):
            for y in range(y_partition):
                pixels += [(upper_left_corner[0] + x, upper_left_corner[1] + y)]
    elif concept == "bricks_top_mid":
        bricks = state_json["bricks"]
        brick_size = (int(bricks[0]['size']['x']), int(bricks[0]['size']['y'])) 
        upper_left_corner = (int(bricks[0]['position']['x']), int(bricks[0]['position']['y']))
        lower_right_corner = (int(bricks[-1]['position']['x']) + brick_size[0] - 1, int(bricks[-1]['position']['y']) + brick_size[1] - 1)

        x_partition = int((lower_right_corner[0] - upper_left_corner[0] + 1)/3)
        y_partition = int((lower_right_corner[1] - upper_left_corner[1] + 1)/2)

        for x in range(x_partition):
            for y in range(y_partition):
                pixels += [(upper_left_corner[0] + x_partition + x, upper_left_corner[1] + y)]
    elif concept == "bricks_top_right":
        bricks = state_json["bricks"]
        brick_size = (int(bricks[0]['size']['x']), int(bricks[0]['size']['y'])) 
        upper_left_corner = (int(bricks[0]['position']['x']), int(bricks[0]['position']['y']))
        lower_right_corner = (int(bricks[-1]['position']['x']) + brick_size[0] - 1, int(bricks[-1]['position']['y']) + brick_size[1] - 1)

        x_partition = int((lower_right_corner[0] - upper_left_corner[0] + 1)/3)
        y_partition = int((lower_right_corner[1] - upper_left_corner[1] + 1)/2)

        for x in range(x_partition):
            for y in range(y_partition):
                pixels += [(upper_left_corner[0] + 2*x_partition + x, upper_left_corner[1] + y)]
    elif concept == "bricks_bottom_left":
        bricks = state_json["bricks"]
        brick_size = (int(bricks[0]['size']['x']), int(bricks[0]['size']['y'])) 
        upper_left_corner = (int(bricks[0]['position']['x']), int(bricks[0]['position']['y']))
        lower_right_corner = (int(bricks[-1]['position']['x']) + brick_size[0] - 1, int(bricks[-1]['position']['y']) + brick_size[1] - 1)

        x_partition = int((lower_right_corner[0] - upper_left_corner[0] + 1)/3)
        y_partition = int((lower_right_corner[1] - upper_left_corner[1] + 1)/2)

        for x in range(x_partition):
            for y in range(y_partition):
                pixels += [(upper_left_corner[0] + x, upper_left_corner[1] + y_partition + y)]
    elif concept == "bricks_bottom_mid":
        bricks = state_json["bricks"]
        brick_size = (int(bricks[0]['size']['x']), int(bricks[0]['size']['y'])) 
        upper_left_corner = (int(bricks[0]['position']['x']), int(bricks[0]['position']['y']))
        lower_right_corner = (int(bricks[-1]['position']['x']) + brick_size[0] - 1, int(bricks[-1]['position']['y']) + brick_size[1] - 1)

        x_partition = int((lower_right_corner[0] - upper_left_corner[0] + 1)/3)
        y_partition = int((lower_right_corner[1] - upper_left_corner[1] + 1)/2)

        for x in range(x_partition):
            for y in range(y_partition):
                pixels += [(upper_left_corner[0] + x_partition + x, upper_left_corner[1] + y_partition + y)]
    elif concept == "bricks_bottom_right":
        bricks = state_json["bricks"]
        brick_size = (int(bricks[0]['size']['x']), int(bricks[0]['size']['y'])) 
        upper_left_corner = (int(bricks[0]['position']['x']), int(bricks[0]['position']['y']))
        lower_right_corner = (int(bricks[-1]['position']['x']) + brick_size[0] - 1, int(bricks[-1]['position']['y']) + brick_size[1] - 1)

        x_partition = int((lower_right_corner[0] - upper_left_corner[0] + 1)/3)
        y_partition = int((lower_right_corner[1] - upper_left_corner[1] + 1)/2)

        for x in range(x_partition):
            for y in range(y_partition):
                pixels += [(upper_left_corner[0] + 2*x_partition + x, upper_left_corner[1] + y_partition + y)]

    #ensure that pixels are not out of scope
    for pixel in pixels:
        if (pixel[0] >= size[0] or pixel[0] <= 0) or (pixel[1] >= size[1] or pixel[1] <= 0):
            print(pixel[0], pixel[1])
            print(size[0], pixel[1])
            pixels.remove(pixel)

    return pixels

def apply_interventions(concept, a_logits, tb, state_json, env, model, pixels, num_samples):
    global INTERVENTIONS

    CF_imp_concept = []
    CF_IV_intensity = []
    if "bricks" in concept:
        concept = "bricks"
    interventions = INTERVENTIONS[concept]

    for i in range(num_samples):
        IV_a_logits = []
        CF_intensity = []
        CF_imp = []
        #get a_logits from interventions
        for IV in interventions:
            logits, intensity = IV(tb, state_json, env, model, pixels)
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
    actions, value, _, _, a_logits = model.step(obs)
    tb.write_state_json(state_json) 

    return list(a_logits), move_distance

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
    actions, value, _, _, a_logits = model.step(obs)
    tb.write_state_json(state_json) 

    return list(a_logits), delta_velocity

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
    actions, value, _, _, a_logits = model.step(obs)
    tb.write_state_json(state_json) 

    return list(a_logits), move_distance

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
    actions, value, _, _, a_logits = model.step(obs)
    tb.write_state_json(state_json) 

    return list(a_logits), num_dead_bricks

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
    actions, value, _, _, a_logits = model.step(obs)
    tb.write_state_json(state_json) 

    return list(a_logits), remove_bricks_depth

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
    actions, value, _, _, a_logits = model.step(obs)
    tb.write_state_json(state_json) 

    return list(a_logits), remove_row

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
    parser.add_argument('-a', '--alg', help='algorithm used for training')
    parser.add_argument('-l', '--load_path', help='path to load the model from')
    parser.add_argument('-hp', '--history_path', help='path of history of a executed episode')
    args = parser.parse_args()

    compute_importance(args.env_name, args.alg, args.load_path, args.history_path, args.num_samples)
