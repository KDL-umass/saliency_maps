# Saliency-Maps

This repository build various saliency maps for models trained using OpenAI's baselines repository. Before starting to create saliency maps, clone https://github.com/akanksha95/baselines.git in the same directory as this repo. This version of baselines is a "fork" of the original baselines repository with code changes to build different saliency maps.

In order to build saliency maps from [Visualizing and Understanding Atari Agents](https://arxiv.org/abs/1711.00138) (Greydanus et al.), type the following command:

`./start_python -m baselines.run --alg=a2c --env=[env-name] --num_timesteps=0 --load_path=[path to model] --visualize True --num_env=1`
