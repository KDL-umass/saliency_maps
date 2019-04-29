# Saliency-Maps

This repository build various saliency maps for models trained using OpenAI's baselines repository. Before starting to create saliency maps, clone https://github.com/akanksha95/baselines.git in the same directory as this repo. This version of baselines is a "fork" of the original baselines repository with code changes to build different saliency maps.

In order to build saliency videos from [Visualizing and Understanding Atari Agents](https://arxiv.org/abs/1711.00138) (Greydanus et al.) for an baselines A2C agent, execute the following command from the directory containing this repo and the baselines repo:

`./start_python -m saliency_maps.visualize_atari.make_movie --env_name=BreakoutToyboxNoFrameskip-v4 --alg=a2c --load_path=[trained model location]`

To run the saliency vs counterfactual importance experiment, run the following:

`./start_python -m saliency_maps.experiments.CFimportance_breakout --env_name=BreakoutToyboxNoFrameskip-v4 --alg=a2c --load_path=[trained model location] --history_path=[pkl file containing history from visualization]`