from baselines import run
from baselines.common import atari_wrappers

import numpy as np
from scipy.misc import imresize # preserves single-pixel info _unlike_ img = img[::2,::2]

#go through one episode and return history[obs (ins), a_logits, values, action (out)]
def rollout(model, env, max_ep_len=3e3):
    history = {'ins': [], 'a_logits': [], 'values': [], 'actions': [], 'color_frame': []}
    episode_length, epr, done = 0, 0, False

    #logger.log("Running trained model")
    print("Running trained model")
    obs = env.reset()
    turtle = atari_wrappers.get_turtle(env)
    # This is a hack to get the starting screen, which throws an error in ALE for amidar
    num_steps = -1

    while not done and episode_length <= max_ep_len:
        episode_length += 1
        actions, value, _, _, a_logits = model.step(obs)
        num_lives = turtle.ale.lives()
        obs, reward, done, info = env.step(actions)
        epr += reward[0]
        color_frame = turtle.toybox.get_rgb_frame()
        #time.sleep(1.0/60.0)

        #print(a_logits, value, actions)

        #save info
        history['ins'].append(obs)
        history['a_logits'].append(a_logits)
        history['values'].append(value)
        history['actions'].append(actions[0])
        history['color_frame'].append(color_frame)
        print('\tstep # {}, reward {:.0f}'.format(episode_length, epr), end='\r')

    return history
