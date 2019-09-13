import pickle

if __name__ == '__main__':

    load_dir = "./saliency_maps/movies/a2c/AmidarToyboxNoFrameskip-v4/perturbation/"
    history_paths = ["default-{}-amidartoyboxnoframeskip-v4-{}.pkl", "IVnonChangingScores-{}-amidartoyboxnoframeskip-v4-{}.pkl", \
                    "IVmultModifyScoresRand-{}-amidartoyboxnoframeskip-v4-{}.pkl", "IVmultModifyScores-{}-amidartoyboxnoframeskip-v4-{}.pkl", \
                    "IVdecrementScore-{}-amidartoyboxnoframeskip-v4-{}.pkl"]

    histories = []
    for path in history_paths:
        paths = []
        for i in range(1, 51):
            path_ = path.format(1000, i)
            history_path = load_dir + path_
            with open(history_path, "rb") as output_file:
                paths.append(pickle.load(output_file))
        histories.append(paths)

    for i,history_type in enumerate(histories):
        print(history_paths[i])
        count_1000_frames = 0
        for j, history in enumerate(history_type):
            print(len(history['state_json']))
            if len(history['state_json']) >= 1000:
                count_1000_frames += 1 
        print(history_paths[i], count_1000_frames)