import os

#NOTE: comment out the creation of MP4 files in make_movie.py when running this. 

default = "python -m saliency_maps.visualize_atari.make_movie --env_name=AmidarToyboxNoFrameskip-v4 \
        --alg=a2c --load_path=./models/AmidarToyboxNoFrameskip-v4/amidar4e7_a2c.model --num_frames=1000 --radius=2"

IVdecrementScore = "python -m saliency_maps.visualize_atari.make_movie --env_name=AmidarToyboxNoFrameskip-v4 \
        --alg=a2c --load_path=./models/AmidarToyboxNoFrameskip-v4/amidar4e7_a2c.model --num_frames=1000 --radius=2 \
        --history_file=default-1000-amidartoyboxnoframeskip-v4-{}.pkl --IVdecrementScore=True"

IVnonChangingScores = "python -m saliency_maps.visualize_atari.make_movie --env_name=AmidarToyboxNoFrameskip-v4 \
        --alg=a2c --load_path=./models/AmidarToyboxNoFrameskip-v4/amidar4e7_a2c.model --num_frames=1000 --radius=2 \
        --history_file=default-1000-amidartoyboxnoframeskip-v4-{}.pkl --IVnonChangingScores=True"

IVmultModifyScores = "python -m saliency_maps.visualize_atari.make_movie --env_name=AmidarToyboxNoFrameskip-v4 \
        --alg=a2c --load_path=./models/AmidarToyboxNoFrameskip-v4/amidar4e7_a2c.model --num_frames=1000 --radius=2 \
        --history_file=default-1000-amidartoyboxnoframeskip-v4-{}.pkl --IVmultModifyScores=True"

#run interventions for each of 50 samples of default file
for i in range(1,51): 
    #generate raw files
    os.system(default)

for i in range(1,51):
    #run IV decrement score
    os.system(IVdecrementScore.format(i))

    #run IV nonchanging scores
    os.system(IVnonChangingScores.format(i))

    #run IV modifying scores
    os.system(IVmultModifyScores.format(i))