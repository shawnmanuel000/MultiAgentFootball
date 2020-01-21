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
parser.add_argument("--model", type=str, default="ddpg", choices=["ddqn", "ddpg", "ppo", "rand"], help="Which reinforcement learning algorithm to use")
parser.add_argument("--steps", type=int, default=100000, help="Number of steps to train the agent")
args = parser.parse_args()

gym_envs = ["CartPole-v0", "MountainCar-v0", "Acrobot-v1", "Pendulum-v0", "MountainCarContinuous-v0", "CarRacing-v0", "BipedalWalker-v2", "BipedalWalkerHardcore-v2", "LunarLander-v2", "LunarLanderContinuous-v2"]
gfb_envs = ["academy_empty_goal_close", "1_vs_1_easy", "5_vs_5", "11_vs_11_stochastic"]
env_name = gfb_envs[2]

def make_env(env_name=env_name, log=False):
	if env_name in gym_envs: return gym.make(env_name)
	reps = ["pixels", "pixels_gray", "extracted", "simple115"]
	env = ggym.create_environment(env_name=env_name, representation=reps[3], logdir='/football/logs/', render=False)
	env.unwrapped.spec = gym.envs.registration.EnvSpec(env_name + "-v0", max_episode_steps=env.unwrapped._config._scenario_cfg.game_duration)
	if log: print(f"State space: {env.observation_space.shape} \nAction space: {env.action_space.n}")
	return env

class AsyncAgent(RandomAgent):
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
		if hasattr(self.agent, "network"): self.agent.network.save_model(dirname, name)

def run(model, steps=10000, ports=16, eval_at=1000, checkpoint=False):
	num_envs = len(ports) if type(ports) == list else min(ports, 64)
	envs = EnvManager(make_env, ports) if type(ports) == list else EnsembleEnv(make_env, ports)
	agent = AsyncAgent(envs.state_size, envs.action_size, num_envs, model)
	logger = Logger(model, env_name, num_envs=num_envs, state_size=agent.stack.state_size, action_size=envs.action_size, action_space=envs.env.action_space)
	states = envs.reset()
	total_rewards = []
	for s in range(steps):
		agent.reset(num_envs)
		env_actions, actions, states = agent.get_env_action(envs.env, states)
		next_states, rewards, dones, _ = envs.step(env_actions)
		agent.train(states, actions, next_states, rewards, dones)
		states = next_states
		if dones[0]:
			rollouts = [rollout(envs.env, agent.reset(1)) for _ in range(5)]
			test_reward = np.mean(rollouts) - np.std(rollouts)
			total_rewards.append(test_reward)
			if checkpoint: agent.save_model(env_name, "checkpoint")
			if env_name in gfb_envs and total_rewards[-1] >= max(total_rewards): agent.save_model(env_name)
			logger.log(f"Step: {s}, Reward: {test_reward+np.std(rollouts):.4f} [{np.std(rollouts):.2f}], Avg: {np.mean(total_rewards):.4f} ({agent.agent.eps:.3f})")

if __name__ == "__main__":
	model = DDPGAgent if args.model == "ddpg" else PPOAgent if args.model == "ppo" else DDQNAgent if args.model == "ddqn" else RandomAgent
	if args.selfport is not None:
		EnvWorker(args.selfport, make_env).start()
	else:
		if len(args.workerports) == 1: args.workerports = args.workerports[0]
		run(model, args.steps, args.workerports)
	print(f"Training finished")