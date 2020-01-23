# MultiAgentFootball

This repository was forked from google-research/football and applies Deep Reinforcement Learning to train agents to play football.

## Setup

To install the Google Football environment, follow the instructions in ./football/README.md which may involve changing to the football directory and running 
`pip3 install .`

Then install the packages for the repository in the root directory by running
`pip3 install -r requirements.txt`

## Folder structure

This repository is organized into 5 folders:
* football - Contains the source code for the Google Football environment
* logs - Contains lists of rewards vs step counts for training different RL * models with visual plots in logs/graphs
* models - Python classes for each RL model (Eg. DDPG, DDQN, PPO)
* saved_models - PyTorch checkpoints for each model for different environments
* utils - Other Python classes and helper code

## Training

To train a RL model, there are two approaches.
1. Train synchronously by running `python3 -B train.py --model [ddqn|ddpg|ppo|rand] --steps NUMBER_OF_STEPS
2. Train asynchronously by running `bash train.sh NUMBER_OF_STEPS [ddqn|ddpg|ppo|rand]

To change the environment to train on, set env_name in train.py accordingly
