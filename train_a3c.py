import os
import gym
import torch
import argparse
import numpy as np
from collections import deque
from models.ppo import PPOAgent
from models.rand import RandomAgent
from models.ddpg import DDPGAgent, EPS_MIN
from utils.envs import EnsembleEnv, EnvManager, EnvWorker, WorldModel, ImgStack
from utils.misc import Logger, rollout

parser = argparse.ArgumentParser(description="A3C Trainer")
parser.add_argument("--workerports", type=int, default=[16], nargs="+", help="The list of worker ports to connect to")
parser.add_argument("--selfport", type=int, default=None, help="Which port to listen on (as a worker server)")
parser.add_argument("--iternum", type=int, default=-1, choices=[-1,0,1], help="Whether to train using World Model to load (0 or 1) or raw images (-1)")
parser.add_argument("--model", type=str, default="ppo", choices=["ddpg", "ppo"], help="Which reinforcement learning algorithm to use")
parser.add_argument("--runs", type=int, default=1, help="Number of episodes to train the agent")
parser.add_argument("--trial", action="store_true", help="Whether to show a trial run training on the Pendulum-v0 environment")
args = parser.parse_args()

ENV_NAME = "CarRacing-v0"

class WorldACAgent(RandomAgent):
	def __init__(self, action_size, num_envs, acagent, statemodel=WorldModel, load="", gpu=True, train=True):
		super().__init__(action_size)
		self.world_model = statemodel(action_size, num_envs, load=load, gpu=gpu)
		self.acagent = acagent(self.world_model.state_size, action_size, load="" if train else load, gpu=gpu)

	def get_env_action(self, env, state, eps=None, sample=True):
		state, latent = self.world_model.get_state(state)
		env_action, action = self.acagent.get_env_action(env, state, eps, sample)
		self.world_model.step(latent, env_action)
		return env_action, action, state

	def train(self, state, action, next_state, reward, done):
		next_state = self.world_model.get_state(next_state)[0]
		self.acagent.train(state, action, next_state, reward, done)

	def reset(self, num_envs=None):
		num_envs = self.world_model.num_envs if num_envs is None else num_envs
		self.world_model.reset(num_envs, restore=False)
		return self

	def save_model(self, dirname="pytorch", name="best"):
		self.acagent.network.save_model(dirname, name)

	def load(self, dirname="pytorch", name="best"):
		self.world_model.load_model(dirname, name)
		self.acagent.network.load_model(dirname, name)
		return self

def run(model, statemodel, runs=1, load_dir="", ports=16):
	num_envs = len(ports) if type(ports) == list else min(ports, 16)
	logger = Logger(model, load_dir, statemodel=statemodel, num_envs=num_envs)
	envs = EnvManager(ENV_NAME, ports) if type(ports) == list else EnsembleEnv(ENV_NAME, ports)
	agent = WorldACAgent(envs.action_size, num_envs, model, statemodel, load=load_dir)
	total_rewards = []
	for ep in range(runs):
		states = envs.reset()
		agent.reset(num_envs)
		total_reward = 0
		for _ in range(envs.env.spec.max_episode_steps):
			env_actions, actions, states = agent.get_env_action(envs.env, states)
			next_states, rewards, dones, _ = envs.step(env_actions, render=(ep%runs==0))
			agent.train(states, actions, next_states, rewards, dones)
			total_reward += np.mean(rewards)
			states = next_states
		rollouts = [rollout(envs.env, agent.reset(1)) for _ in range(10)]
		test_reward = np.mean(rollouts) - np.std(rollouts)
		total_rewards.append(test_reward)
		agent.save_model(load_dir, "checkpoint")
		if total_rewards[-1] >= max(total_rewards): agent.save_model(load_dir)
		logger.log(f"Ep: {ep}, Reward: {total_reward:.4f}, Test: {test_reward+np.std(rollouts):.4f} [{np.std(rollouts):.2f}], Avg: {np.mean(total_rewards):.4f} ({agent.acagent.eps:.3f})")
	envs.close()

def trial(model, steps=40000, ports=16):
	env_name = "Pendulum-v0"
	envs = EnvManager(ENV_NAME, ports) if type(ports) == list else EnsembleEnv(ENV_NAME, ports)
	agent = model(envs.state_size, envs.action_size, decay=0.99)
	env = gym.make(env_name)
	state = envs.reset()
	test_rewards = []
	for s in range(steps):
		env_action, action = agent.get_env_action(env, state)
		next_state, reward, done, _ = envs.step(env_action)
		agent.train(state, action, next_state, reward, done)
		state = next_state
		if s % env.spec.max_episode_steps == 0:
			test_reward = np.mean([rollout(env, agent) for _ in range(10)])
			test_rewards.append(test_reward)
			print(f"Ep: {s//env.spec.max_episode_steps}, Rewards: {test_reward}, Avg: {np.mean(test_rewards)}")
			if test_reward > -200: break
	env.close()
	envs.close()

if __name__ == "__main__":
	dirname = "pytorch" if args.iternum < 0 else f"iter{args.iternum}/"
	state = ImgStack if args.iternum < 0 else WorldModel
	model = PPOAgent if args.model == "ppo" else DDPGAgent
	if args.trial:
		trial(model, ports=args.workerports)
	elif args.selfport is not None:
		EnvWorker(args.selfport, ENV_NAME).start()
	else:
		if len(args.workerports) == 1: args.workerports = args.workerports[0]
		run(model, state, args.runs, dirname, args.workerports)