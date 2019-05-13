import os

#NOTE: comment out the creation of MP4 files in make_movie.py when running this. 
#run interventions for each of 50 samples of default file
for i in range(5,56): 
    #run IV decrement score
    os.system("./start_python -m saliency_maps.visualize_atari.make_movie --env_name=AmidarToyboxNoFrameskip-v4 \
        --alg=a2c --load_path=./models/AmidarToyboxNoFrameskip-v4/amidar4e7_a2c.model --num_frames=150 --radius=2 \
        --history_file=default-150-amidartoyboxnoframeskip-v4-{}.pkl --IVdecrementScore=True".format(i))

    #run IV nonchanging scores
    os.system("./start_python -m saliency_maps.visualize_atari.make_movie --env_name=AmidarToyboxNoFrameskip-v4 \
        --alg=a2c --load_path=./models/AmidarToyboxNoFrameskip-v4/amidar4e7_a2c.model --num_frames=150 --radius=2 \
        --history_file=default-150-amidartoyboxnoframeskip-v4-{}.pkl --IVnonChangingScores=True".format(i))

    #run IV modifying scores
    os.system("./start_python -m saliency_maps.visualize_atari.make_movie --env_name=AmidarToyboxNoFrameskip-v4 \
        --alg=a2c --load_path=./models/AmidarToyboxNoFrameskip-v4/amidar4e7_a2c.model --num_frames=150 --radius=2 \
        --history_file=default-150-amidartoyboxnoframeskip-v4-{}.pkl --IVmultModifyScores=True".format(i))