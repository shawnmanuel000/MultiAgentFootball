import gym
import numpy as np
import gfootball.env as ggym
from collections import deque
from models.ppo import PPOAgent
from models.rand import RandomAgent
from models.ddqn import DDQNAgent
from models.ddpg import DDPGAgent, EPS_MIN
from utils.envs import EnsembleEnv, EnvManager, EnvWorker, ImgStack, RawStack
from utils.misc import rollout

def make_env(env_name="academy_empty_goal_close", log=False):
	#env = gym.make("CartPole-v0")
	reps = ["pixels", "pixels_gray", "extracted", "simple115"]
	env = ggym.create_environment(env_name=env_name, representation=reps[3], logdir='/football/logs/', render=False)
	if log: print(f"State space: {env.observation_space.shape} \nAction space: {env.action_space.n}")
	return env

def trial(steps=1000, ports=16):
	env = make_env(log=True)
	envs = EnsembleEnv(make_env, ports)
	stack = RawStack(envs.state_size, ports)
	agent = PPOAgent(stack.state_size, envs.action_size)
	state = stack.get_state(envs.reset())
	test_rewards = []
	for s in range(steps):
		env_action, action = agent.get_env_action(env, state)
		next_state, reward, done, _ = envs.step(env_action)
		next_state = stack.get_state(next_state)
		agent.train(state, action, next_state, reward, done)
		state = next_state
		if done[0]:
			test_reward = np.mean([rollout(env, agent) for _ in range(1)])
			test_rewards.append(test_reward)
			print(f"Step: {s}, Rewards: {test_reward}, Avg: {np.mean(test_rewards)}")
	env.close()
	envs.close()

trial()