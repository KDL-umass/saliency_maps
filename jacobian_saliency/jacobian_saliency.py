import tensorflow as tf
from tensorflow.python.ops.parallel_for.gradients import jacobian
import numpy as np
import pickle
import time
from scipy.misc import imresize

from baselines.common.input import observation_placeholder
from baselines.common.tf_util import adjust_shape

import matplotlib 
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as manimation

from vis.visualization import visualize_saliency

from saliency_maps.visualize_atari.saliency import get_env_meta, score_frame, saliency_on_atari_frame, occlude
from saliency_maps.visualize_atari.make_movie import setUp

from tensorflow.python.ops.parallel_for.gradients import jacobian

def run_through_model(model, obs, mode='actor'):
    _, value, _, _, a_logits, grads = model.step(obs)
    return grads

def saliency_on_atari_frame(saliency, atari, fudge_factor, channel=2, sigma=0):
    # sometimes saliency maps are a bit clearer if you blur them
    # slightly...sigma adjusts the radius of that blur
    pmax = np.max(saliency)
    S = np.zeros((110, 84))
    S[18:102, :] = saliency
    S = imresize(saliency, size=[atari.shape[0],atari.shape[1]], interp='bilinear').astype(np.float32)
    S = S if sigma == 0 else gaussian_filter(S, sigma=sigma)
    S -= np.min(S)
    S = fudge_factor * pmax * S / np.max(S)

    I = atari.astype(np.uint16)
    I[:,:,channel] += S.astype(np.uint16)
    I = np.clip(I, 1., 255.).astype(np.uint8)
    return I

def make_movie(alg, env_name, num_frames, prefix, load_history_path, load_model_path, resolution=75, first_frame=1):
    # set up env and model
    env, model = setUp(env_name, alg, load_model_path)
    ob_space = env.observation_space

    with open(load_history_path, "rb") as input_file:
        history = pickle.load(input_file)

    save_dir = "./saliency_maps/movies/{}/{}/jacobian/".format(alg, env_name)
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

                # get input that get color_frame
                obs = history['ins'][ix-1]

                jacobian = run_through_model(model, obs)
                frame = saliency_on_atari_frame((jacobian[0,:,:,3]**2).squeeze(), frame, fudge_factor=10, channel=2) #blue

                plt.imshow(frame) ; plt.title(env_name.lower(), fontsize=15)
                writer.grab_frame() ; f.clear()
                
                tstr = time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start))
                print('\ttime: {} | progress: {:.1f}%'.format(tstr, 100*i/min(num_frames, total_frames)), end='\r')
    print('\nfinished.')


# make_movie("a2c", "BreakoutToyboxNoFrameskip-v4", 150, "jacobian_default", "./saliency_maps/movies/a2c/BreakoutToyboxNoFrameskip-v4/perturbation/default-150-breakouttoyboxnoframeskip-v4-6.pkl", "./models/BreakoutToyboxNoFrameskip-v4/breakout4e7_a2c.model")
make_movie("a2c", "AmidarToyboxNoFrameskip-v4", 250, "jacobian_default", "./saliency_maps/movies/a2c/AmidarToyboxNoFrameskip-v4/perturbation/default-250-amidartoyboxnoframeskip-v4-2.pkl", "./models/AmidarToyboxNoFrameskip-v4/amidar4e7_a2c.model")

