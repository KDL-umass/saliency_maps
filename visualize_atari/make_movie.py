from saliency_maps.visualize_atari.saliency import get_env_meta, score_frame, saliency_on_atari_frame, occlude
from saliency_maps.rollout import *

from baselines.run import train, build_env

import matplotlib.pyplot as plt
import matplotlib as mpl ; mpl.use("Agg")
import matplotlib.animation as manimation

import gym, os, sys, time, argparse, pickle

def make_movie(env_name, alg, env, model, load_path, num_frames=20, first_frame=0, resolution=75, save_dir=None, \
                density=5, radius=5, prefix='default', IVmoveball=False, IVsymbricks=False, IVmodifyScore=False, \
                IVmultModifyScores=False, IVnonChangingScores=False, IVdecrementScore=False, IVmoveEnemies=False):
    print('making movie using model at ', load_path)
    # set up env and model
    if env is None or model is None:
        env, model = setUp(env_name, alg, load_path)

    # generate file names and save_dir
    if save_dir is None:
        save_dir = "./saliency_maps/movies/{}/{}/perturbation/".format(alg, env_name)
    movie_title = "{}-{}-{}-{}.mp4".format(prefix, num_frames, env_name.lower(), 1)
    history_title = "{}-{}-{}-{}.pkl".format(prefix, num_frames, env_name.lower(), 1)
    movie_title, history_title = update_filenames(save_dir, movie_title, history_title)

    # get a rollout of the policy
    max_ep_len = first_frame + num_frames + 1
    history = rollout(model, env, max_ep_len=max_ep_len)
    #env.close()
    
    # save history
    filehandler = open(save_dir + history_title, 'wb') 
    pickle.dump(history, filehandler)

    # make movie!
    _make_movie(env_name, model, movie_title, history, num_frames, first_frame, resolution, \
                save_dir, density, radius, prefix)

    if IVmoveball:
        make_intervention_movie(env_name, alg, env, model, load_path, history_title, max_ep_len, num_frames, first_frame, resolution, \
                                save_dir, density, radius, IVmoveball=True)
    elif IVmodifyScore:
        make_intervention_movie(env_name, alg, env, model, load_path, history_title, max_ep_len, num_frames, first_frame, resolution, \
                                save_dir, density, radius, IVmodifyScore=True)
    elif IVmultModifyScores:
        make_intervention_movie(env_name, alg, env, model, load_path, history_title, max_ep_len, num_frames, first_frame, resolution, \
                                save_dir, density, radius, IVmultModifyScores=True)
    elif IVsymbricks:
        make_intervention_movie(env_name, alg, env, model, load_path, history_title, max_ep_len, num_frames, first_frame, resolution, \
                                save_dir, density, radius, IVsymbricks=True)
    elif IVnonChangingScores:
        make_intervention_movie(env_name, alg, env, model, load_path, history_title, max_ep_len, num_frames, first_frame, resolution, \
                                save_dir, density, radius, IVnonChangingScores=True)
    elif IVdecrementScore:
        make_intervention_movie(env_name, alg, env, model, load_path, history_title, max_ep_len, num_frames, first_frame, resolution, \
                                save_dir, density, radius, IVdecrementScore=True)
    elif IVmoveEnemies:
        make_intervention_movie(env_name, alg, env, model, load_path, history_title, max_ep_len, num_frames, first_frame, resolution, \
                                save_dir, density, radius, IVmoveEnemies=True)

    return env, model

def make_intervention_movie(env_name, alg, env, model, load_path, history_file, max_ep_len=3e3, num_frames=20, \
                            first_frame=0, resolution=75, save_dir=None, density=5, radius=5, IVmoveball=False, IVsymbricks=False, \
                            IVmodifyScore=False, IVmultModifyScores=False, IVnonChangingScores=False, IVdecrementScore=False, \
                            IVmoveEnemies=False):

    if env is None or model is None:
        env, model = setUp(env_name, alg, load_path)

    if IVmoveball:
        print('making movie with IVmoveball intervention using model at ', load_path)
        prefix = "IVmoveball"
        if save_dir is None:
            save_dir = "./saliency_maps/movies/{}/{}/perturbation/".format(alg, env_name)

        # get interventional history
        default_history_file = open(save_dir + history_file, 'rb') 
        default_history = pickle.load(default_history_file)
        history = single_intervention_move_ball(model, env, default_history, move_distance=5, max_ep_len=max_ep_len)

    if IVsymbricks:
        print('making movie with IVsymbricks intervention using model at ', load_path)
        prefix = "IVsymbricks"
        if save_dir is None:
            save_dir = "./saliency_maps/movies/{}/{}/perturbation/".format(alg, env_name)

        # get interventional history
        default_history_file = open(save_dir + history_file, 'rb') 
        default_history = pickle.load(default_history_file)
        history = single_intervention_symmetric_brick(model, env, default_history, max_ep_len=max_ep_len, intervene_step=120)

    if IVmodifyScore:
        print('making movie with IVmodifyScore intervention using model at ', load_path)
        prefix = "IVmodifyScore"
        if save_dir is None:
            save_dir = "./saliency_maps/movies/{}/{}/perturbation/".format(alg, env_name)

        # get interventional history
        default_history_file = open(save_dir + history_file, 'rb') 
        default_history = pickle.load(default_history_file)
        history = single_intervention_modify_score(model, env, default_history, max_ep_len=max_ep_len, intervene_step=80)

    if IVmultModifyScores:
        print('making movie with IVmultModifyScores intervention using model at ', load_path)
        prefix = "IVmultModifyScoresRand"
        if save_dir is None:
            save_dir = "./saliency_maps/movies/{}/{}/perturbation/".format(alg, env_name)

        # get interventional history
        default_history_file = open(save_dir + history_file, 'rb') 
        default_history = pickle.load(default_history_file)
        history = multiple_intervention_modify_score(model, env, default_history, max_ep_len=max_ep_len, random_score=True)

    if IVnonChangingScores:
        print('making movie with IVnonChangingScores intervention using model at ', load_path)
        prefix = "IVnonChangingScores"
        if save_dir is None:
            save_dir = "./saliency_maps/movies/{}/{}/perturbation/".format(alg, env_name)

        # get interventional history
        default_history_file = open(save_dir + history_file, 'rb') 
        default_history = pickle.load(default_history_file)
        history = multiple_intervention_nonchanging_score(model, env, default_history, max_ep_len=max_ep_len)

    if IVdecrementScore:
        print('making movie with IVdecrementScore intervention using model at ', load_path)
        prefix = "IVdecrementScore"
        if save_dir is None:
            save_dir = "./saliency_maps/movies/{}/{}/perturbation/".format(alg, env_name)

        # get interventional history
        default_history_file = open(save_dir + history_file, 'rb') 
        default_history = pickle.load(default_history_file)
        history = multiple_intervention_decrement_score(model, env, default_history, max_ep_len=200)

    if IVmoveEnemies:
        print('making movie with IVmoveEnemies intervention using model at ', load_path)
        prefix = "IVmoveEnemies"
        if save_dir is None:
            save_dir = "./saliency_maps/movies/{}/{}/perturbation/".format(alg, env_name)

        # get interventional history
        default_history_file = open(save_dir + history_file, 'rb') 
        default_history = pickle.load(default_history_file)
        history = multiple_intervention_move_enemies(model, env, default_history)

    # generate file names
    movie_title = "{}-{}-{}-{}.mp4".format(prefix, num_frames, env_name.lower(), history_file.split(".pkl")[0].split("-")[-1])
    history_title = "{}-{}-{}-{}.pkl".format(prefix, num_frames, env_name.lower(), history_file.split(".pkl")[0].split("-")[-1])
    print(history_title)

    # save history
    filehandler = open(save_dir + history_title, 'wb') 
    pickle.dump(history, filehandler)

    # make movie!
    _make_movie(env_name, model, movie_title, history, num_frames, first_frame, resolution, \
            save_dir, density, radius, prefix)

    return env, model

def _make_movie(env_name, model, movie_title, history, num_frames=20, first_frame=1, resolution=75, \
                save_dir='./saliency_maps/movies/', density=5, radius=5, prefix='default'):
    # set up dir variables and environment
    load_dir = env_name.lower()
    meta = get_env_meta(env_name)

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
                actor_saliency = score_frame(model, history, ix, radius, density, interp_func=occlude, mode='actor')
                critic_saliency = score_frame(model, history, ix, radius, density, interp_func=occlude, mode='critic')
            
                frame = saliency_on_atari_frame(actor_saliency, frame, fudge_factor=meta['actor_ff'], channel=2) #blue
                frame = saliency_on_atari_frame(critic_saliency, frame, fudge_factor=meta['critic_ff'], channel=0) #red

                plt.imshow(frame) ; plt.title(env_name.lower(), fontsize=15)
                writer.grab_frame() ; f.clear()
                
                tstr = time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start))
                print('\ttime: {} | progress: {:.1f}%'.format(tstr, 100*i/min(num_frames, total_frames)), end='\r')
    print('\nfinished.')

def update_filenames(directory, movie_file, history_file):
    if os.path.isfile(directory + movie_file):
        count = 1
        while True:
            count += 1
            new_movie_file = movie_file.split(".mp4")[0][:-1] + str(count) + ".mp4"
            new_history_file = history_file.split(".pkl")[0][:-1] + str(count) + ".pkl"
            if os.path.isfile(directory + new_history_file):
                continue
            else:
                movie_file = new_movie_file
                history_file = new_history_file
                break

    return movie_file, history_file

def setUp(env, alg, load_path):
    args = Bunch({'env':env, 'alg':alg, 'num_timesteps':0, 'seed':None, 'num_env':1, 'network':None})
    extra_args = {'load_path':load_path}

    model, env = train(args, extra_args)
    env.close()
    env = build_env(args, extra_args)

    return env, model

class Bunch(object):
  def __init__(self, adict):
    self.__dict__.update(adict)

# user might also want to access make_movie function from some other script
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-e', '--env_name', default='BreakoutToyboxNoFrameskip-v4', type=str, help='name of gym environment')
    parser.add_argument('-a', '--alg', help='algorithm used for training')
    parser.add_argument('-l', '--load_path', help='path to load the model from')
    parser.add_argument('-d', '--density', default=5, type=int, help='density of grid of gaussian blurs')
    parser.add_argument('-r', '--radius', default=5, type=int, help='radius of gaussian blur')
    parser.add_argument('-f', '--num_frames', default=20, type=int, help='number of frames in movie')
    parser.add_argument('-i', '--first_frame', default=0, type=int, help='index of first frame')
    parser.add_argument('-dpi', '--resolution', default=75, type=int, help='resolution (dpi)')
    parser.add_argument('-ns', '--num_samples', default=1, type=int, help='number of samples to run for')
    parser.add_argument('-s', '--save_dir', default=None, type=str, help='dir to save agent logs and checkpoints')
    parser.add_argument('-p', '--prefix', default='default', type=str, help='prefix to help make video name unique')
    parser.add_argument('-hp', '--history_file', default=None, type=str, help='location of history to do intervention')
    parser.add_argument('-imb', '--IVmoveball', default=False, type=bool, help='intervention move ball in breakout')
    parser.add_argument('-imsb', '--IVsymbricks', default=False, type=bool, help='intervention modify symmetry of bricks in breakout')
    parser.add_argument('-ims', '--IVmodifyScore', default=False, type=bool, help='intervention change score mid-game in amidar')
    parser.add_argument('-imms', '--IVmultModifyScores', default=False, type=bool, help='intervention change scores mid-game in amidar')
    parser.add_argument('-incs', '--IVnonChangingScores', default=False, type=bool, help='intervention non changing scores in amidar')
    parser.add_argument('-imds', '--IVdecrementScore', default=False, type=bool, help='intervention decrement scores at each timestep in amidar')
    parser.add_argument('-ime', '--IVmoveEnemies', default=False, type=bool, help='intervention move enemy on same protocol at multiple steps in amidar')
    args = parser.parse_args()

    env, model = None, None
    for i in range(args.num_samples):
        if args.history_file is None:
            env, model = make_movie(args.env_name, args.alg, env, model, args.load_path, args.num_frames, args.first_frame, args.resolution,
                args.save_dir, args.density, args.radius, args.prefix, args.IVmoveball, args.IVsymbricks, 
                args.IVmodifyScore, args.IVmultModifyScores, args.IVnonChangingScores, args.IVdecrementScore)
        else:
            env, model = make_intervention_movie(args.env_name, args.alg, env, model, args.load_path, args.history_file, num_frames=args.num_frames, 
                first_frame=args.first_frame, resolution=args.resolution, save_dir=args.save_dir, density=args.density, 
                radius=args.radius, IVmoveball=args.IVmoveball, IVsymbricks=args.IVsymbricks, IVmodifyScore=args.IVmodifyScore, 
                IVmultModifyScores=args.IVmultModifyScores, IVnonChangingScores=args.IVnonChangingScores, IVdecrementScore=args.IVdecrementScore,
                IVmoveEnemies=args.IVmoveEnemies)
