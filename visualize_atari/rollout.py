from baselines.common import atari_wrappers

from toybox.interventions.breakout import *
from toybox.toybox import Toybox, Input

import numpy as np
from scipy.misc import imresize # preserves single-pixel info _unlike_ img = img[::2,::2]
import random

#go through one episode and return history[obs (ins), a_logits, values, action (out), rewards, color_frame and state_json]
def rollout(model, env, max_ep_len=3e3):
    history = {'ins': [], 'a_logits': [], 'values': [], 'actions': [], 'rewards': [], 'color_frame': [], 'state_json': []}
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
        color_frame = tb.get_rgb_frame()
        state_json = tb.state_to_json()

        #print(a_logits, value, actions)

        #save info
        history['ins'].append(obs)
        history['a_logits'].append(a_logits)
        history['values'].append(value)
        history['actions'].append(actions[0])
        history['rewards'].append(epr)
        history['color_frame'].append(color_frame)
        history['state_json'].append(state_json)
        print('\tstep # {}, reward {:.0f}'.format(episode_length, epr), end='\r')

    return history

def single_intervention_move_ball(model, env, rollout_history, max_ep_len=3e3, move_distance=1, intervene_step=20):
    history = {'ins': [], 'a_logits': [], 'values': [], 'actions': [], 'rewards': [], 'color_frame': [], 'state_json': []}
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
        color_frame = tb.get_rgb_frame()
        state_json = tb.state_to_json()

        #save info
        history['ins'].append(obs)
        history['a_logits'].append(rollout_history['a_logits'][episode_length])
        history['values'].append(rollout_history['values'][episode_length])
        history['actions'].append(rollout_history['actions'][episode_length])
        history['rewards'].append(epr)
        history['color_frame'].append(color_frame)
        history['state_json'].append(state_json)

        episode_length += 1

    print("Intervening on ball now and forward simulating")
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
        obs, _, _, _ = env.step(0)

    while not done and episode_length <= max_ep_len:
        episode_length += 1
        actions, value, _, _, a_logits = model.step(obs)
        num_lives = turtle.ale.lives()
        obs, reward, done, info = env.step(actions)
        epr += reward[0]
        color_frame = tb.get_rgb_frame()
        state_json = tb.state_to_json()

        #save info
        history['ins'].append(obs)
        history['a_logits'].append(a_logits)
        history['values'].append(value)
        history['actions'].append(actions[0])
        history['color_frame'].append(color_frame)
        history['rewards'].append(epr)
        history['state_json'].append(state_json)
        print('\tstep # {}, reward {:.0f}'.format(episode_length, epr), end='\r')

    return history

def single_intervention_symmetric_brick(model, env, rollout_history, max_ep_len=3e3, intervene_step=20):
    history = {'ins': [], 'a_logits': [], 'values': [], 'actions': [], 'rewards': [], 'color_frame': [], 'state_json': []}
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
        color_frame = tb.get_rgb_frame()
        state_json = tb.state_to_json()

        #save info
        history['ins'].append(obs)
        history['a_logits'].append(rollout_history['a_logits'][episode_length])
        history['values'].append(rollout_history['values'][episode_length])
        history['actions'].append(rollout_history['actions'][episode_length])
        history['rewards'].append(epr)
        history['color_frame'].append(color_frame)
        history['state_json'].append(state_json)

        episode_length += 1

    print("Intervening on bricks now and forward simulating")
    #subtract (240-12) - x.pos of alive bricks
    with BreakoutIntervention(tb) as intervention: 
        bricks = intervention.get_bricks()
        bricks_to_flip = []
        for i,brick in enumerate(bricks):
            if brick['alive'] is False:
                intervention.set_brick(i)
                sym_xPos = (240-12) - brick['position']['x']
                #print(sym_xPos)
                for j,brick2 in enumerate(bricks):
                    if brick2['position']['x'] == sym_xPos and brick2['position']['y'] == brick['position']['y']:
                        #print(brick2)
                        bricks_to_flip.append(j)
                        break
                #print(brick)

        #print(bricks_to_flip)
        for brick_index in bricks_to_flip:
            intervention.set_brick(brick_index, alive=False)

    #forward simulate 3 steps with no-op action
    for i in range(3):
        obs, _, _, _ = env.step(0)

    while not done and episode_length <= max_ep_len:
        episode_length += 1
        actions, value, _, _, a_logits = model.step(obs)
        num_lives = turtle.ale.lives()
        obs, reward, done, info = env.step(actions)
        epr += reward[0]
        color_frame = tb.get_rgb_frame()
        state_json = tb.state_to_json()
        #time.sleep(1.0/60.0)

        #save info
        history['ins'].append(obs)
        history['a_logits'].append(a_logits)
        history['values'].append(value)
        history['actions'].append(actions[0])
        history['rewards'].append(epr)
        history['color_frame'].append(color_frame)
        history['state_json'].append(state_json)
        print('\tstep # {}, reward {:.0f}'.format(episode_length, epr), end='\r')

    return history

def single_intervention_modify_score(model, env, rollout_history, max_ep_len=3e3, abs_score=0, intervene_step=20):
    history = {'ins': [], 'a_logits': [], 'values': [], 'actions': [], 'rewards': [], 'color_frame': [], 'state_json': []}
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
        color_frame = tb.get_rgb_frame()
        state_json = tb.state_to_json()

        #save info
        history['ins'].append(obs)
        history['a_logits'].append(rollout_history['a_logits'][episode_length])
        history['values'].append(rollout_history['values'][episode_length])
        history['actions'].append(rollout_history['actions'][episode_length])
        history['rewards'].append(epr)
        history['color_frame'].append(color_frame)
        history['state_json'].append(state_json)

        episode_length += 1

    amidar_modify_score(tb, rollout_history, episode_length, abs_score, env)

    while not done and episode_length <= max_ep_len:
        episode_length += 1
        actions, value, _, _, a_logits = model.step(obs)
        num_lives = turtle.ale.lives()
        obs, reward, done, info = env.step(actions)
        epr += reward[0]
        color_frame = tb.get_rgb_frame()
        state_json = tb.state_to_json()

        #save info
        history['ins'].append(obs)
        history['a_logits'].append(a_logits)
        history['values'].append(value)
        history['actions'].append(actions[0])
        history['rewards'].append(epr)
        history['color_frame'].append(color_frame)
        history['state_json'].append(state_json)
        print('\tstep # {}, reward {:.0f}'.format(episode_length, epr), end='\r')

    return history

def multiple_intervention_modify_score(model, env, rollout_history, max_ep_len=3e3, abs_score=0, intervene_steps=[20,40,80,100,120,140], random_score=False):
    history = {'ins': [], 'a_logits': [], 'values': [], 'actions': [], 'rewards': [], 'color_frame': [], 'state_json': []}
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

    while episode_length < intervene_steps[0]:
        obs, reward, done, info = env.step(rollout_history['actions'][episode_length])
        epr += reward[0]
        color_frame = tb.get_rgb_frame()
        state_json = tb.state_to_json()

        #save info
        history['ins'].append(obs)
        history['a_logits'].append(rollout_history['a_logits'][episode_length])
        history['values'].append(rollout_history['values'][episode_length])
        history['actions'].append(rollout_history['actions'][episode_length])
        history['rewards'].append(epr)
        history['color_frame'].append(color_frame)
        history['state_json'].append(state_json)

        episode_length += 1

    amidar_modify_score(tb, rollout_history, episode_length, abs_score, env, random_score)

    while not done and episode_length <= max_ep_len:
        episode_length += 1
        actions, value, _, _, a_logits = model.step(obs)
        num_lives = turtle.ale.lives()
        obs, reward, done, info = env.step(actions)
        epr += reward[0]
        color_frame = tb.get_rgb_frame()
        state_json = tb.state_to_json()

        #intervene
        if episode_length in intervene_steps:
            amidar_modify_score(tb, rollout_history, episode_length, abs_score, env)

        #save info
        history['ins'].append(obs)
        history['a_logits'].append(a_logits)
        history['values'].append(value)
        history['actions'].append(actions[0])
        history['rewards'].append(epr)
        history['color_frame'].append(color_frame)
        history['state_json'].append(state_json)
        print('\tstep # {}, reward {:.0f}'.format(episode_length, epr), end='\r')

    return history

def multiple_intervention_nonchanging_score(model, env, rollout_history, max_ep_len=3e3, abs_score=0):
    history = {'ins': [], 'a_logits': [], 'values': [], 'actions': [], 'rewards': [], 'color_frame': [], 'state_json': []}
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

    while not done and episode_length <= max_ep_len:
        episode_length += 1
        actions, value, _, _, a_logits = model.step(obs)
        num_lives = turtle.ale.lives()
        obs, reward, done, info = env.step(actions)
        epr += reward[0]
        state_json = tb.state_to_json()

        #intervene on score
        state_json['score'] = abs_score
        tb.write_state_json(state_json)

        color_frame = tb.get_rgb_frame()

        #save info
        history['ins'].append(obs)
        history['a_logits'].append(a_logits)
        history['values'].append(value)
        history['actions'].append(actions[0])
        history['rewards'].append(epr)
        history['color_frame'].append(color_frame)
        history['state_json'].append(state_json)
        print('\tstep # {}, reward {:.0f}'.format(episode_length, epr), end='\r')

    return history

def multiple_intervention_decrement_score(model, env, rollout_history, max_ep_len=3e3, max_score=200):
    history = {'ins': [], 'a_logits': [], 'values': [], 'actions': [], 'rewards': [], 'color_frame': [], 'state_json': []}
    episode_length, epr, done = 0, 0, False

    #logger.log("Running trained model")
    print("Running trained model")
    obs = env.reset()
    turtle = atari_wrappers.get_turtle(env)
    tb = turtle.toybox

    #start new game and set start state to the same state as original game with max_score
    tb.new_game()
    state_json = rollout_history['state_json'][0]
    state_json['score'] = max_score
    tb.write_state_json(state_json)
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
        state_json = tb.state_to_json()

        #intervene on score
        state_json['score'] -= 1
        tb.write_state_json(state_json)

        color_frame = tb.get_rgb_frame()

        #save info
        history['ins'].append(obs)
        history['a_logits'].append(a_logits)
        history['values'].append(value)
        history['actions'].append(actions[0])
        history['rewards'].append(epr)
        history['color_frame'].append(color_frame)
        history['state_json'].append(state_json)
        print('\tstep # {}, reward {:.0f}'.format(episode_length, epr), end='\r')

    return history

def amidar_modify_score(tb, rollout_history, index, abs_score, env, random_score=False):
    print("Intervening on score now and forward simulating")
    print("old: ", tb.state_to_json()['score'])

    new_state = rollout_history['state_json'][index]
    if random_score:
        new_state['score'] = random.randint(1,201)
    else:   
        new_state['score'] = abs_score
    tb.write_state_json(new_state)

    print("new: ", tb.state_to_json()['score'])
    #forward simulate 3 steps with no-op action
    for i in range(3):
        obs, _, _, _ = env.step(0)
