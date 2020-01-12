import gym
import argparse
import numpy as np
import gfootball.env as ggym
from collections import deque
from models.ppo import PPOAgent
from models.ddqn import DDQNAgent
from models.ddpg import DDPGAgent
from models.rand import RandomAgent
from utils.envs import EnsembleEnv, EnvManager, EnvWorker, ImgStack, RawStack
from utils.misc import Logger, rollout

parser = argparse.ArgumentParser(description="A3C Trainer")
parser.add_argument("--workerports", type=int, default=[16], nargs="+", help="The list of worker ports to connect to")
parser.add_argument("--selfport", type=int, default=None, help="Which port to listen on (as a worker server)")
parser.add_argument("--model", type=str, default="ddqn", choices=["ddqn", "ddpg", "ppo"], help="Which reinforcement learning algorithm to use")
parser.add_argument("--steps", type=int, default=1000, help="Number of steps to train the agent")
args = parser.parse_args()

envs = ["11_vs_11_stochastic", "academy_empty_goal_close"]
env_name = envs[1]

def make_env(env_name=env_name, log=False):
	reps = ["pixels", "pixels_gray", "extracted", "simple115"]
	env = ggym.create_environment(env_name=env_name, representation=reps[3], logdir='/football/logs/', render=False)
	if log: print(f"State space: {env.observation_space.shape} \nAction space: {env.action_space.n}")
	return env

class PixelAgent(RandomAgent):
	def __init__(self, state_size, action_size, num_envs, agent, load="", gpu=True, train=True):
		super().__init__(state_size, action_size)
		statemodel = RawStack if len(state_size) == 1 else ImgStack
		self.stack = statemodel(state_size, num_envs, load=load, gpu=gpu)
		self.agent = agent(self.stack.state_size, action_size, load="" if train else load, gpu=gpu)

	def get_env_action(self, env, state, eps=None, sample=True):
		state = self.stack.get_state(state)
		env_action, action = self.agent.get_env_action(env, state, eps, sample)
		return env_action, action, state

	def train(self, state, action, next_state, reward, done):
		next_state = self.stack.get_state(next_state)
		self.agent.train(state, action, next_state, reward, done)

	def reset(self, num_envs=None):
		num_envs = self.stack.num_envs if num_envs is None else num_envs
		self.stack.reset(num_envs, restore=False)
		return self

	def save_model(self, dirname="pytorch", name="best"):
		self.agent.network.save_model(dirname, name)

	def load(self, dirname="pytorch", name="best"):
		self.stack.load_model(dirname, name)
		self.agent.network.load_model(dirname, name)
		return self

def run(model, steps=10000, ports=16, eval_at=1000):
	num_envs = len(ports) if type(ports) == list else min(ports, 64)
	logger = Logger(model, env_name, num_envs=num_envs)
	envs = EnvManager(make_env, ports) if type(ports) == list else EnsembleEnv(make_env, ports)
	agent = PixelAgent(envs.state_size, envs.action_size, num_envs, model)
	states = envs.reset()
	total_rewards = []
	for s in range(steps):
		agent.reset(num_envs)
		env_actions, actions, states = agent.get_env_action(envs.env, states)
		next_states, rewards, dones, _ = envs.step(env_actions)
		agent.train(states, actions, next_states, rewards, dones)
		states = next_states
		if s % eval_at == 0:
			rollouts = [rollout(envs.env, agent.reset(5)) for _ in range(1)]
			test_reward = np.mean(rollouts) - np.std(rollouts)
			total_rewards.append(test_reward)
			agent.save_model(env_name, "checkpoint")
			if total_rewards[-1] >= max(total_rewards): agent.save_model(env_name)
			logger.log(f"Ep: {s//eval_at}, Reward: {test_reward+np.std(rollouts):.4f} [{np.std(rollouts):.2f}], Avg: {np.mean(total_rewards):.4f} ({agent.agent.eps:.3f})")

if __name__ == "__main__":
	model = DDPGAgent if args.model == "ddpg" else PPOAgent if args.model == "ppo" else DDQNAgent
	if args.selfport is not None:
		EnvWorker(args.selfport, make_env).start()
	else:
		if len(args.workerports) == 1: args.workerports = args.workerports[0]
		run(model, args.steps, args.workerports)
	print(f"Training finished")