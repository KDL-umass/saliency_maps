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

X = None
# TODO: implement jacobian saliency
'''
def jacobian(model, layer, top_dh, X):
    global top_h_ ; top_h_ = None
    def hook_top_h(m, i, o): global top_h_ ; top_h_ = o.clone()
    hook1 = layer.register_forward_hook(hook_top_h)
    _ = model(X) # do a forward pass so the forward hooks can be called

    # backprop positive signal
    torch.autograd.backward(top_h_, top_dh.clone(), retain_variables=True) # backward hooks are called here
    hook1.remove()
    return X[0].grad.data.clone().numpy(), X[0].data.clone().numpy()
'''
def jacobian_(model, inp, out):
    grads = visualize_saliency(model, -1, filter_indices=0, seed_input=inp)
    print(grads)
    return grads

def jacobian_tf(model, inp, out):    
    jacobian_matrix = np.zeros((len(out),len(out)))
    inp_ = tf.cast(inp, tf.float32) / 255.
    print(inp_)
    for m in range(len(out)):
        # We iterate over the M elements of the output vector
        #grad_func = tf.gradients(tf.convert_to_tensor(out[m]), tf.convert_to_tensor(inp))[0]
        grad_func = tf.gradients(tf.convert_to_tensor(out[m]), inp_)[0]
        if grad_func is None: grad_func = tf.convert_to_tensor(0)
        gradients = model.step_model._evaluate([grad_func], inp)
        #print(gradients)
        jacobian_matrix[m][m] = gradients[0]
    
    print(jacobian_matrix)
    jacobian_matrix = imresize(jacobian_matrix, size=[84,84], interp='bilinear').astype(np.float32)
    return np.array(jacobian_matrix)

def js(model, inp, output):
    global X
    print("obs before processing: ", inp.shape)
    inp = adjust_shape(X, inp)
    inp_ = tf.cast(inp, tf.float32) / 255.
    print("obs after processing: ", inp_.shape)

    # for f in [0,1,2,3]: #because atari passes 4 frames per round
    #     print(f)
    #     print("shape of single image: ", obs[0,:,:,f].shape)
    #     plt.imshow(obs[0,:,:,f])
    #     plt.savefig("figure_{}".format(f))

    # max_outp = tf.constant(max(output))

    # max_outp = tf.reduce_max(output)

    # max_outp = tf.convert_to_tensor(output.reshape(1,-1))

    # max_outp = tf.reduce_max(output, 1)

    output = output.reshape(1,-1)
    u = tf.random_uniform(tf.shape(output))
    x = tf.argmax(output - tf.log(-tf.log(u)), axis=-1)
    max_outp = tf.reduce_max(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=x))

    # x = tf.placeholder(tf.float32, shape=inp_.shape)

    print(max_outp, inp_)
    saliency_op = tf.gradients(max_outp, inp_) #try "stop_gradients=inp_" as parameter
    print(saliency_op)
    # if saliency_op is None: saliency_op = tf.convert_to_tensor(0)
    # gradients = model.step_model._evaluate([saliency_op], inp)
    # print(gradients)
    # return gradients
    return [0]

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

    #actor_jacobian, _ = jacobian(model, model.actor_linear, top_dh_actor, obs)

    #state.grad.mul_(0) ; X = (state, (hx, cx))
    #critic_jacobian, _ = jacobian(model, model.critic_linear, top_dh_critic, obs)

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
                # derivative is simply the output policy distribution
                out_actor = history['a_logits'][ix]
                out_critic = history['values'][ix]

                # get input
                obs = history['ins'][ix-1]
                #print(obs)
                #print(out_actor)

                actor_jacobian = js(model, obs, out_actor)
                critic_jacobian = js(model, obs, out_critic)
                print(actor_jacobian)
                print(critic_jacobian)
                '''
                if actor_jacobian[0] == None: actor_jacobian = np.array([0])
                if critic_jacobian[0] == None: critic_jacobian = np.array([0])
                print(actor_jacobian)
                print(critic_jacobian)
                '''
            
                # frame = saliency_on_atari_frame((actor_jacobian**2).squeeze(), frame, fudge_factor=1, channel=2) #blue
                # frame = saliency_on_atari_frame((critic_jacobian**2).squeeze(), frame, fudge_factor=15, channel=0) #red

                # plt.imshow(frame) ; plt.title(env_name.lower(), fontsize=15)
                # writer.grab_frame() ; f.clear()
                
                tstr = time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start))
                print('\ttime: {} | progress: {:.1f}%'.format(tstr, 100*i/min(num_frames, total_frames)), end='\r')
    print('\nfinished.')


make_movie("a2c", "BreakoutToyboxNoFrameskip-v4", 25, "jacobian_default", "./saliency_maps/movies/a2c/BreakoutToyboxNoFrameskip-v4/default-150-breakouttoyboxnoframeskip-v4-6.pkl", "./models/BreakoutToyboxNoFrameskip-v4/breakout4e7_a2c.model")

