# Saliency-Maps

This repository contains code from experiments discussed in our ICLR 2020 paper [Exploratory Not Explanatory: Counterfactual Analysis of Saliency Maps for Deep RL](https://openreview.net/forum?id=rkl3m1BFDB).

It includes resources for generating saliency maps for deep reinforcement learning (RL) models and additionally contains experiments to empirically examine the causal relationships between saliency and agent behavior. It also provides implementations of three types of saliency maps used in RL: (1) [Jacobian](https://arxiv.org/abs/1511.06581), (2) [perturbation-based](https://arxiv.org/abs/1711.00138), and (3) [object-based](https://arxiv.org/abs/1809.06061).

If you use this code or are inspired by our methodology, please cite our [ICLR paper](https://openreview.net/pdf?id=rkl3m1BFDB):
```
@inproceedings{atrey2020exploratory,
  title={{Exploratory Not Explanatory: Counterfactual Analysis of Saliency Maps for Deep RL}},
  author={Atrey, Akanksha and Clary, Kaleigh and Jensen, David},
  booktitle={{International Conference on Learning Representations (ICLR)}},
  year={2020}
}
```
Please direct all queries to **Akanksha Atrey** (aatrey at cs dot umass dot edu) or [open an issue in this repository](https://github.com/KDL-umass/saliency_maps/issues/new).

## About

**Abstract:** Saliency maps are often used to suggest explanations of the behavior of deep reinforcement learning (RL) agents. However, the explanations derived from saliency maps are often unfalsifiable and can be highly subjective. We introduce an empirical approach grounded in counterfactual reasoning to test the hypotheses generated from saliency maps and show that explanations suggested by saliency maps are often not supported by experiments. Our experiments suggest that saliency maps are best viewed as an exploratory tool rather than an explanatory tool.

## Setup

### Python
This repository requires Python 3 (>=3.5).

### Toybox Repository
We use [Toybox](https://arxiv.org/abs/1812.02850), a set of fully parameterized implementation of Atari games, to generate interventional data under counterfactual conditions. Visit the [Toybox repository](https://github.com/jjfiv/toybox) and follow the setup instructions. The `saliency_maps` repository should reside in the `toybox/ctoybox` folder within the Toybox repository.

### Baselines Repository
All agents used in this work are trained using the OpenAI's baselines implementation. Clone [this version](https://github.com/akanksha95/baselines.git) of the baselines repository in the same directory as this repository and follow the setup instructions (in the `toybox/ctoybox` folder). This version of baselines is a fork of the original baselines repository with code changes to accomodate building different saliency maps.

## Training Deep RL Agents on Toybox
We use the [A2C algorithm](https://towardsdatascience.com/understanding-actor-critic-methods-931b97b6df3f) for all our experiments. 

To train a deep RL model on Amidar using the A2C algorithm, execute the following command:

`python3 -m baselines.run --alg=a2c --env=AmidarToyboxNoFrameskip-v4 --num_timesteps=4e7 --save_path=toybox/ctoybox/models/amidar4e7_a2c.model`

To train a deep RL model on Breakout using the A2C algorithm, execute the following command:

`python3 -m baselines.run --alg=a2c --env=BreakoutToyboxNoFrameskip-v4 --num_timesteps=4e7 --save_path=toybox/ctoybox/models/breakout4e7_a2c.model`

## Building Saliency Maps

The implementation follows that all three types of saliency videos are created for a single episode. The perturbation saliency video must be created first before creating the object and Jacobian saliency maps. When the perturbation saliency video is created, it simultaneously creates an associated pickle file with the actions chosen by the agent. This pickle file will be used when creating object and Jacobian saliency videos to avoid discrepancies in agent behavior.

To build a perturbation saliency video on a Breakout agent, execute the following command from the `toybox/ctoybox` directory:

`python3 -m saliency_maps.visualize_atari.make_movie --env_name=BreakoutToyboxNoFrameskip-v4 --alg=a2c --load_path=toybox/ctoybox/models/breakout4e7_a2c.model`

To build a object saliency video on a Breakout agent, execute the following command from the `toybox/ctoybox` directory:

`python3 -m saliency_maps.object_saliency.object_saliency --env_name=BreakoutToyboxNoFrameskip-v4 --alg=a2c --load_path=toybox/ctoybox/models/breakout4e7_a2c.model --history_path=[location of pkl file of actions]`

To build a Jacobian saliency video on a Breakout agent, execute the following command from the `toybox/ctoybox` directory:

`python3 -m saliency_maps.jacobian_saliency.jacobian_saliency.make_movie --env_name=BreakoutToyboxNoFrameskip-v4 --alg=a2c --load_model_path=toybox/ctoybox/models/breakout4e7_a2c.model --load_history_path=[location of pkl file of actions]`

### Interventions

All interventions can be found in `saliency_maps/rollout.py`. These functions are accessible as boolean parameters when building perturbation saliency maps based on input parameters. Note, these functions can only be called once a default agent has been run and a corresponding pickle file has been generated. The set of interventions is not exhaustive. Users can build their own interventions in the same python file and add corresponding edits in the `saliency_maps/visualize_atari/make_movie()` file.

The generic command to create videos of interventions is: 

`python3 -m saliency_maps.visualize_atari.make_movie --env_name=BreakoutToyboxNoFrameskip-v4 --alg=a2c --load_path=toybox/ctoybox/models/breakout4e7_a2c.model --history_file [location of pkl file of actions from default run] --IVmoveball=True, --IVsymbricks=True, --IVmodifyScore=True, --IVmultModifyScores=True, --IVnonChangingScores=True, --IVdecrementScore=True, --IVmoveEnemies=True, --IVmoveEnemiesBack=True, --IVshiftBricks=True`

All interventions can be run at the same time by setting their parameter value to be True.

### Experiments in Paper

Code to run experiments from the ICLR paper and corresponding figures can be found in `saliency_maps/experiments`.

