from baselines.common import atari_wrappers

from toybox.interventions.breakout import *
from toybox.toybox import Toybox, Input

import numpy as np
from scipy.misc import imresize # preserves single-pixel info _unlike_ img = img[::2,::2]

#go through one episode and return history[obs (ins), a_logits, values, action (out)]
def rollout(model, env, max_ep_len=3e3):
    history = {'ins': [], 'a_logits': [], 'values': [], 'actions': [], 'color_frame': [], 'state_json': []}
    episode_length, epr, done = 0, 0, False

    #logger.log("Running trained model")
    print("Running trained model")
    obs = env.reset()
    turtle = atari_wrappers.get_turtle(env)
    tb = turtle.toybox
    start_state_json = tb.state_to_json()
    history['state_json'].append(start_state_json)
    # This is a hack to get the starting screen, which throws an error in ALE for amidar
    num_steps = -1

    while not done and episode_length <= max_ep_len:
        episode_length += 1
        actions, value, _, _, a_logits = model.step(obs)
        num_lives = turtle.ale.lives()
        obs, reward, done, info = env.step(actions)
        epr += reward[0]
        color_frame = turtle.toybox.get_rgb_frame()
        state_json = tb.state_to_json()
        #time.sleep(1.0/60.0)

        #print(a_logits, value, actions)

        #save info
        history['ins'].append(obs)
        history['a_logits'].append(a_logits)
        history['values'].append(value)
        history['actions'].append(actions[0])
        history['color_frame'].append(color_frame)
        history['state_json'].append(state_json)
        print('\tstep # {}, reward {:.0f}'.format(episode_length, epr), end='\r')

    return history

def single_intervention_move_ball(model, env, rollout_history, max_ep_len=3e3, move_distance=1, intervene_step=20):
    history = {'ins': [], 'a_logits': [], 'values': [], 'actions': [], 'color_frame': [], 'state_json': []}
    episode_length, epr, done = 0, 0, False

    #logger.log("Running trained model")
    print("Running trained model")
    obs = env.reset()
    turtle = atari_wrappers.get_turtle(env)
    tb = turtle.toybox

    #start new game and set start state to the same state as original game
    tb.new_game()
    tb.write_state_json(rollout_history['state_json'][0])
    start_state_json = tb.state_to_json()
    history['state_json'].append(start_state_json)

    # This is a hack to get the starting screen, which throws an error in ALE for amidar
    num_steps = -1

    while episode_length < intervene_step:
        obs, reward, done, info = env.step(rollout_history['actions'][episode_length])
        epr += reward[0]
        color_frame = turtle.toybox.get_rgb_frame()
        state_json = tb.state_to_json()

        #save info
        history['ins'].append(obs)
        history['a_logits'].append(rollout_history['a_logits'][episode_length])
        history['values'].append(rollout_history['values'][episode_length])
        history['actions'].append(rollout_history['actions'][episode_length])
        history['color_frame'].append(color_frame)
        history['state_json'].append(state_json)

        episode_length += 1

    print("Intervening now and forward simulating")
    with BreakoutIntervention(tb) as intervention: 
        ball_pos = intervention.get_ball_position()
        print("old: ", ball_pos)
        ball_pos['x'] = ball_pos['x'] + move_distance
        ball_pos['y'] = ball_pos['y'] + move_distance
        print("new: ", ball_pos)
        intervention.set_ball_position(ball_pos)
        ball_pos_post = intervention.get_ball_position()
        assert ball_pos_post['x'] == ball_pos['x']

        #forward simulate 3 steps with no-op action
        for i in range(3):
            tb.apply_action(Input())

    while not done and episode_length <= max_ep_len:
        episode_length += 1
        actions, value, _, _, a_logits = model.step(obs)
        num_lives = turtle.ale.lives()
        obs, reward, done, info = env.step(actions)
        epr += reward[0]
        color_frame = turtle.toybox.get_rgb_frame()
        state_json = tb.state_to_json()
        #time.sleep(1.0/60.0)

        #print(a_logits, value, actions)

        #save info
        history['ins'].append(obs)
        history['a_logits'].append(a_logits)
        history['values'].append(value)
        history['actions'].append(actions[0])
        history['color_frame'].append(color_frame)
        history['state_json'].append(state_json)
        print('\tstep # {}, reward {:.0f}'.format(episode_length, epr), end='\r')

    return history