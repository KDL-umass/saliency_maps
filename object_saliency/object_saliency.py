from saliency_maps.object_saliency import *
import tensorflow as tf
import numpy as np
import pickle, argparse, time
import cv2 as cv
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

def get_objPixels(env_objects, object_template, env_name, env, frame, state_json):
    #using template matching to get objects as in Iyer et al. -- see https://docs.opencv.org/master/d4/dc6/tutorial_py_template_matching.html
    threshold = 0.8
    pixels = []
    source = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    #template matching
    for obj in env_objects:
        print("concept: ", obj)
        template = cv.imread(object_template[obj], 0)
        w, h = template.shape[::-1]

        res = cv.matchTemplate(source,template,cv.TM_CCOEFF_NORMED)
        loc = np.where(res >= threshold)

        for pt in zip(*loc[::-1]):
            pix_obj = []
            for x in range(w):
                for y in range(h):
                    pix_obj.append((pt[0]+x, pt[1]+y))
            pixels.append(pix_obj)
            # cv.rectangle(frame, pt, (pt[0] + w, pt[1] + h), (0,0,255), 1) #check object detection by adding rectangular frame

    #manual pixel collection
    if "Breakout" in env_name:
        pixels.append(get_concept_pixels_breakout('balls', state_json, [frame.shape[1],frame.shape[0]]))
        pixels.append(get_concept_pixels_breakout('paddle', state_json, [frame.shape[1],frame.shape[0]]))
        pixels.append(get_concept_pixels_breakout('score', state_json, [frame.shape[1],frame.shape[0]]))
        pixels.append(get_concept_pixels_breakout('lives', state_json, [frame.shape[1],frame.shape[0]]))
        pixels += get_concept_pixels_breakout('bricks', state_json, [frame.shape[1],frame.shape[0]])
    elif "Amidar" in env_name:
        turtle = atari_wrappers.get_turtle(env)
        tb = turtle.toybox
        tb.write_state_json(state_json)
        pixels.append(get_concept_pixels_amidar('score', state_json, [frame.shape[1],frame.shape[0]], tb))
        pixels.append(get_concept_pixels_amidar('lives', state_json, [frame.shape[1],frame.shape[0]], tb))

    return pixels

def run_through_model(model, obs, mode='actor'):
    _, value, _, _, a_logits = model.step(obs)

    if mode == 'actor':
        return a_logits
    elif mode == 'critic':
        return value

def score_frame(env_name, env, model, history, ix, obj_pixels, mode='actor'):
    orig_obs = history['ins'][ix]
    q = run_through_model(model, orig_obs, mode=mode) #without masking objects

    #get pixels of objects and modify previous 3 frames with moving objects (i.e. first 2 objects in breakout (balls + paddle) and first 6 objects in amidar (i.e. player and enemies))
    obj_pixels.pop(0)
    if "Breakout" in env_name:
        obj_pixels.append(get_objPixels(BREAKOUT_OBJECT_KEYS, BREAKOUT_OBJECT_TEMPLATES, env_name, env, history['color_frame'][ix], history['state_json'][ix]))
        for f in [0,1,2]:
            obj_pixels[f] = obj_pixels[f][:2] + obj_pixels[3][2:]
    elif "Amidar" in env_name:
        obj_pixels.append(get_objPixels(AMIDAR_OBJECT_KEYS, AMIDAR_OBJECT_TEMPLATES, env_name, env, history['color_frame'][ix], history['state_json'][ix]))
        for f in [0,1,2]:
            obj_pixels[f] = obj_pixels[f][:6] + obj_pixels[3][6:]
    else:
        print("Undefined env_name: neither Breakout nor Amidar.")
        return None

    #mask and calculate score
    len_obj = len(obj_pixels[3])
    scores = np.zeros(len_obj)
    for i,pix in enumerate(range(len_obj)):
        processed_obs = np.copy(orig_obs)
        for f in [0,1,2,3]: #because atari passes 4 frames per round
            processed_obs[0,:,:,f]  = mask_object(orig_obs[0,:,:,f], history['color_frame'][ix], obj_pixels[f][i])
        q_o = run_through_model(model, processed_obs, mode=mode) #with masking object o
        # scores[i] = q - q_o
        scores[i] = np.linalg.norm(q - q_o)
    # print('scores:', scores)

    return scores, obj_pixels

def mask_object(obs, frame, obj_pixels):
    #modify obs (84x84) to be of size frame (PROBLEM: can't mask a 2d image with background color??)
    M = np.zeros((110, 84))
    M[18:102, :] = obs
    M = imresize(obs, size=[frame.shape[0],frame.shape[1]]).astype(np.float32)

    #mask each pixel with background color
    for pixel in obj_pixels:
        M[pixel[1], pixel[0]] = 0

    #resize M to be 84x84
    processed_obs = imresize(M, size=[84,84]).astype(np.float32)

    # plt.imshow(obs)
    # plt.show()
    # plt.imshow(processed_obs)
    # plt.show()

    return processed_obs

def saliency_on_atari_frame(frame, pixels, score):
    S = 128 * np.ones([frame.shape[0],frame.shape[1], 3], dtype=np.uint8) #gray background

    for i,s in enumerate(score):
        for pixel in pixels[i]:
            S[pixel[1], pixel[0]] = S[pixel[1], pixel[0]] + int(s*10)
    S = np.clip(S, a_min=0, a_max=255)

    # plt.imshow(S)
    # plt.show()

    return S

def make_movie(alg, env_name, num_frames, prefix, load_history_path, load_model_path, resolution=75, first_frame=0):
    # set up env and model
    env, model = setUp(env_name, alg, load_model_path)

    with open(load_history_path, "rb") as input_file:
        history = pickle.load(input_file)

    save_dir = "./saliency_maps/movies/{}/{}/object/".format(alg, env_name)
    movie_title = "{}-{}-{}-{}.mp4".format(prefix, num_frames, env_name.lower(), load_history_path.split(".pkl")[0][-1:])

    if 'Breakout' in env_name:
        obj_pixels = [get_objPixels(BREAKOUT_OBJECT_KEYS, BREAKOUT_OBJECT_TEMPLATES, env_name, env, history['color_frame'][first_frame], history['state_json'][first_frame])] * 4
    elif 'Amidar' in env_name:
        obj_pixels = [get_objPixels(AMIDAR_OBJECT_KEYS, AMIDAR_OBJECT_TEMPLATES, env_name, env, history['color_frame'][first_frame], history['state_json'][first_frame])] * 4

    # make the movie!
    start = time.time()
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title=movie_title, artist='aatrey', comment='atari-object-saliency-video')
    writer = FFMpegWriter(fps=8, metadata=metadata)
    
    prog = '' ; total_frames = len(history['ins'])
    f = plt.figure(figsize=[6, 6*1.3], dpi=resolution)
    with writer.saving(f, save_dir + movie_title, resolution):
        for i in range(num_frames):
            ix = first_frame+i
            if ix < total_frames: # prevent loop from trying to process a frame ix greater than rollout length
                frame = history['color_frame'][ix]
                saliency, obj_pixels = score_frame(env_name, env, model, history, ix, obj_pixels, mode='actor')
                frame = saliency_on_atari_frame(frame, obj_pixels[3], saliency)

                plt.imshow(frame) ; plt.title(env_name.lower(), fontsize=15)
                writer.grab_frame() ; f.clear()
                
                tstr = time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start))
                print('\ttime: {} | progress: {:.1f}%'.format(tstr, 100*i/min(num_frames, total_frames)), end='\r')
    print('\nfinished.')

# make_movie("a2c", "BreakoutToyboxNoFrameskip-v4", 25, "object_default", "./saliency_maps/movies/a2c/BreakoutToyboxNoFrameskip-v4/perturbation/default-150-breakouttoyboxnoframeskip-v4-6.pkl", "./models/BreakoutToyboxNoFrameskip-v4/breakout4e7_a2c.model")
# make_movie("a2c", "AmidarToyboxNoFrameskip-v4", 25, "object_default", "./saliency_maps/movies/a2c/AmidarToyboxNoFrameskip-v4/perturbation/default-250-amidartoyboxnoframeskip-v4-2.pkl", "./models/AmidarToyboxNoFrameskip-v4/amidar4e7_a2c.model")

# user might also want to access make_movie function from some other script
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-e', '--env_name', default='BreakoutToyboxNoFrameskip-v4', type=str, help='name of gym environment')
    parser.add_argument('-a', '--alg', help='algorithm used for training')
    parser.add_argument('-l', '--load_path', help='path to load the model from')
    parser.add_argument('-f', '--num_frames', default=25, type=int, help='number of frames in movie')
    parser.add_argument('-i', '--first_frame', default=0, type=int, help='index of first frame')
    parser.add_argument('-dpi', '--resolution', default=75, type=int, help='resolution (dpi)')
    parser.add_argument('-p', '--prefix', default='object_default', type=str, help='prefix to help make video name unique')
    parser.add_argument('-hp', '--history_path', default=None, type=str, help='location of history to do intervention')
    args = parser.parse_args()

    make_movie(args.alg, args.env_name, args.num_frames, args.prefix, args.history_path, args.load_path, args.resolution, args.first_frame)
