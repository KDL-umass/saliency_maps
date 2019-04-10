import matplotlib.pyplot as plt
import matplotlib as mpl ; mpl.use("Agg")
import matplotlib.animation as manimation

import gym, os, sys, time, argparse, pickle

from saliency_maps.visualize_atari.a2c.saliency import get_env_meta, score_frame, saliency_on_atari_frame, occlude
from saliency_maps.visualize_atari.a2c.rollout import rollout, single_intervention_move_ball

def make_movie(env_name, env, model, num_frames=20, first_frame=0, resolution=75, save_dir='./saliency_maps/movies/', \
                density=5, radius=5, prefix='default', IVmoveball=False):
    
    # generate file names
    movie_title = "{}-{}-{}-{}.mp4".format(prefix, num_frames, env_name.lower(), 1)
    history_title = "{}-{}-{}-{}.pkl".format(prefix, num_frames, env_name.lower(), 1)
    movie_title, history_title = update_filenames(save_dir, movie_title, history_title)

    # get a rollout of the policy
    max_ep_len = first_frame + num_frames + 1
    history = rollout(model, env, max_ep_len=max_ep_len)
    
    # save history
    filehandler = open(save_dir + history_title, 'wb') 
    pickle.dump(history, filehandler)

    # make movie!
    _make_movie(env_name, env, model, movie_title, history, num_frames, first_frame, resolution, \
                save_dir, density, radius, prefix)

    if IVmoveball:
        make_intervention_movie(env_name, env, model, history_title, num_frames, first_frame, resolution, \
                                save_dir, density, radius)

def make_intervention_movie(env_name, env, model, default_history_path, num_frames=20, first_frame=0, resolution=75, \
                            save_dir='./saliency_maps/movies/', density=5, radius=5, prefix='IVmoveball', \
                            IVmoveball=True):
    if IVmoveball:
        prefix = "IVmoveball"
        # get interventional history
        default_history_file = open(save_dir + default_history_path, 'rb') 
        default_history = pickle.load(default_history_file)
        history = single_intervention_move_ball(model, env, default_history, move_distance=3, intervene_step=20)

        # generate file names
        movie_title = "{}-{}-{}-{}.mp4".format(prefix, num_frames, env_name.lower(), default_history_path.split(".pkl")[0][-1:])
        history_title = "{}-{}-{}-{}.pkl".format(prefix, num_frames, env_name.lower(), default_history_path.split(".pkl")[0][-1:])

        # save history
        filehandler = open(save_dir + history_title, 'wb') 
        pickle.dump(history, filehandler)

        # make movie!
        _make_movie(env_name, env, model, movie_title, history, num_frames, first_frame, resolution, \
                save_dir, density, radius, prefix)


def _make_movie(env_name, env, model, movie_title, history, num_frames=20, first_frame=0, resolution=75, \
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
            if os.path.isfile(directory + new_movie_file):
                continue
            else:
                movie_file = new_movie_file
                history_file = new_history_file
                break

    return movie_file, history_file

# user might also want to access make_movie function from some other script
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-en', '--env_name', default='BreakoutToyboxNoFrameskip-v4', type=str, help='name of gym environment')
    parser.add_argument('-e', '--env', help='baselines environment object')
    parser.add_argument('-m', '--model', help='baselines model object')
    parser.add_argument('-d', '--density', default=5, type=int, help='density of grid of gaussian blurs')
    parser.add_argument('-r', '--radius', default=5, type=int, help='radius of gaussian blur')
    parser.add_argument('-f', '--num_frames', default=20, type=int, help='number of frames in movie')
    parser.add_argument('-i', '--first_frame', default=150, type=int, help='index of first frame')
    parser.add_argument('-dpi', '--resolution', default=75, type=int, help='resolution (dpi)')
    parser.add_argument('-s', '--save_dir', default='./movies/', type=str, help='dir to save agent logs and checkpoints')
    parser.add_argument('-p', '--prefix', default='default', type=str, help='prefix to help make video name unique')
    args = parser.parse_args()

    make_movie(args.env, args.checkpoint, args.num_frames, args.first_frame, args.resolution,
        args.save_dir, args.density, args.radius, args.prefix, args.overfit_mode)
