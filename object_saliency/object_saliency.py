import tensorflow as tf
import numpy as np
import pickle
import time
from scipy.misc import imresize

from baselines.common.input import observation_placeholder
from baselines.common.tf_util import adjust_shape
from baselines.common import atari_wrappers

import matplotlib 
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as manimation

from saliency_maps.visualize_atari.make_movie import setUp
from saliency_maps.experiments import CONCEPTS
from saliency_maps.experiments.CFimportance_amidar import get_concept_pixels as get_concept_pixels_amidar
from saliency_maps.experiments.CFimportance_breakout import get_concept_pixels as get_concept_pixels_breakout

def run_through_model(model, obs, mode='actor'):
    _, value, _, _, a_logits = model.step(obs)
    # return value if mode == 'critic' else a_logits
    return value

def score_frame(env_name, env, model, history, ix, mode='actor'):
    orig_obs = history['ins'][ix]
    q = run_through_model(model, orig_obs, mode=mode) #without masking objects

    if "Breakout" in env_name:
        objects = CONCEPTS['Breakout']
    elif "Amidar" in env_name:
        objects = CONCEPTS['Amidar']
    else:
        print("Undefined env_name: neither Breakout nor Amidar.")
        return None

    scores = np.zeros(len(objects))
    pixels = []

    for i,o in enumerate(objects):
        processed_obs = np.copy(orig_obs)
        for f in [0,1,2,3]: #because atari passes 4 frames per round
            processed_obs[0,:,:,f], pix = mask_object(orig_obs[0,:,:,f], env_name, env, o, history['state_json'][ix], history['color_frame'][ix])
        print('processed_obs size', processed_obs.shape)
        q_o = run_through_model(model, processed_obs, mode=mode) #with masking object o
        print(q, q_o)
        pixels.append(pix)
        scores[i] = q - q_o
    print('scores:', scores)

    return scores, pixels

def mask_object(obs, env_name, env, obj, state_json, frame):
    #get pixels of obj
    if "Breakout" in env_name:
        pixels = get_concept_pixels_breakout(obj, state_json, [frame.shape[1],frame.shape[0]])
    if "Amidar" in env_name:
        turtle = atari_wrappers.get_turtle(env)
        tb = turtle.toybox
        tb.write_state_json(state_json)
        pixels = get_concept_pixels_amidar(obj, state_json, [frame.shape[1],frame.shape[0]], tb)

    #modify obs (84x84) to be of size frame (PROBLEM: can't mask a 2d image with background color??)
    M = np.zeros((110, 84))
    M[18:102, :] = obs
    M = imresize(obs, size=[frame.shape[0],frame.shape[1]]).astype(np.float32)

    #mask each pixel with background color
    for pixel in pixels:
        M[pixel[1], pixel[0]] = 0

    #resize M to be 84x84
    processed_obs = imresize(M, size=[84,84]).astype(np.float32)

    # plt.imshow(obs)
    # plt.show()
    # plt.imshow(processed_obs)
    # plt.show()

    return processed_obs, pixels

def saliency_on_atari_frame(frame, pixels, score):
    S = 128 * np.ones([frame.shape[0],frame.shape[1], 3], dtype=np.uint8) #gray background

    for i,s in enumerate(score):
        for pixel in pixels[i]:
            S[pixel[1], pixel[0]] += int(s*100)

    plt.imshow(S)
    plt.show()

    return S

def make_movie(alg, env_name, num_frames, prefix, load_history_path, load_model_path, resolution=75, first_frame=1):
    global X 

    # set up env and model
    env, model = setUp(env_name, alg, load_model_path)
    ob_space = env.observation_space
    X = observation_placeholder(ob_space)

    with open(load_history_path, "rb") as input_file:
        history = pickle.load(input_file)

    save_dir = "./saliency_maps/movies/{}/{}/".format(alg, env_name)
    movie_title = "{}-{}-{}-{}.mp4".format(prefix, num_frames, env_name.lower(), load_history_path.split(".pkl")[0][-1:])

    # make the movie!
    start = time.time()
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title=movie_title, artist='greydanus', comment='atari-saliency-video')
    writer = FFMpegWriter(fps=8, metadata=metadata)
    
    prog = '' ; total_frames = len(history['ins'])
    f = plt.figure(figsize=[6, 6*1.3], dpi=resolution)
    with writer.saving(f, save_dir + movie_title, resolution):
        for i in range(num_frames):
            ix = first_frame+i
            if ix < total_frames: # prevent loop from trying to process a frame ix greater than rollout length
                frame = history['color_frame'][ix]
                # actor_saliency = score_frame(env_name, env, model, history, ix, mode='actor')
                critic_saliency, pixels = score_frame(env_name, env, model, history, ix, mode='critic')

                # frame = saliency_on_atari_frame((actor_jacobian**2).squeeze(), frame, fudge_factor=1, channel=2)
                frame = saliency_on_atari_frame(frame, pixels, critic_saliency)

                # plt.imshow(frame) ; plt.title(env_name.lower(), fontsize=15)
                # writer.grab_frame() ; f.clear()
                
                # tstr = time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start))
                # print('\ttime: {} | progress: {:.1f}%'.format(tstr, 100*i/min(num_frames, total_frames)), end='\r')
    print('\nfinished.')

# make_movie("a2c", "BreakoutToyboxNoFrameskip-v4", 25, "object_default", "./saliency_maps/movies/a2c/BreakoutToyboxNoFrameskip-v4/default-150-breakouttoyboxnoframeskip-v4-6.pkl", "./models/BreakoutToyboxNoFrameskip-v4/breakout4e7_a2c.model")
make_movie("a2c", "AmidarToyboxNoFrameskip-v4", 25, "object_default", "./saliency_maps/movies/a2c/AmidarToyboxNoFrameskip-v4/default-250-amidartoyboxnoframeskip-v4-2.pkl", "./models/AmidarToyboxNoFrameskip-v4/amidar4e7_a2c.model")

