import gym
import torch
import pickle
import numpy as np
from collections import deque
from torchvision import transforms
from utils.multiprocess import Manager, Worker
from utils.misc import rgb2gray, resize

FRAME_STACK = 2					# The number of consecutive image states to combine for training a3c on raw images
NUM_ENVS = 16					# The default number of environments to simultaneously train the a3c in parallel

class RawStack():
	def __init__(self, state_size, num_envs=1, stack_len=FRAME_STACK, load="", gpu=True):
		self.state_size = state_size
		self.stack_len = stack_len
		self.reset(num_envs)

	def reset(self, num_envs, restore=False):
		pass

	def get_state(self, state):
		return state

	def step(self, state, env_action):
		pass

class ImgStack():
	def __init__(self, state_size, num_envs=1, stack_len=FRAME_STACK, load="", gpu=True):
		self.process = lambda x: np.expand_dims(np.transpose(resize(rgb2gray(x) if x.shape[-1] == 3 else x), (2,0,1)), 0)
		self.state_size = [*self.process(np.zeros(state_size)).shape[-2:], (1 if state_size[-1]==3 else state_size[-1])*stack_len]
		self.stack_len = stack_len
		self.reset(num_envs)

	def reset(self, num_envs, restore=False):
		self.num_envs = num_envs
		self.stack = deque(maxlen=self.stack_len)

	def get_state(self, state):
		state = np.concatenate([self.process(s) for s in state]) if self.num_envs > 1 else self.process(state)
		while len(self.stack) < self.stack_len: self.stack.append(state)
		self.stack.append(state)
		return np.concatenate(self.stack, axis=1)

	def step(self, state, env_action):
		pass

class EnsembleEnv():
	def __init__(self, make_env, num_envs=NUM_ENVS):
		self.env = make_env()
		self.envs = [make_env() for _ in range(num_envs)]
		self.state_size = [self.env.observation_space.n] if hasattr(self.env.observation_space, "n") else self.env.observation_space.shape
		self.action_size = [self.env.action_space.n] if hasattr(self.env.action_space, "n") else self.env.action_space.shape

	def reset(self):
		states = [env.reset() for env in self.envs]
		return np.stack(states)

	def step(self, actions, render=False):
		results = []
		for env,action in zip(self.envs, actions):
			ob, rew, done, info = env.step(action)
			ob = env.reset() if done else ob
			results.append((ob, rew, done, info))
			if render: env.render()
		obs, rews, dones, infos = zip(*results)
		return np.stack(obs), np.stack(rews), np.stack(dones), infos

	def close(self):
		self.env.close()
		for env in self.envs:
			env.close()

	def __del__(self):
		self.close()

class EnvWorker(Worker):
	def __init__(self, self_port, make_env):
		super().__init__(self_port)
		self.env = make_env()

	def start(self):
		step = 0
		rewards = 0
		while True:
			data = pickle.loads(self.conn.recv(100000))
			if data["cmd"] == "RESET":
				message = self.env.reset()
				rewards = 0
			elif data["cmd"] == "STEP":
				state, reward, done, info = self.env.step(data["item"])
				state = self.env.reset() if done else state
				rewards += reward
				step += 1
				message = (state, reward, done, info)
				if data["render"]: self.env.render()
				if done: 
					print(f"Step: {step}, Reward: {rewards}")
					rewards = 0
			elif data["cmd"] == "CLOSE":
				self.env.close()
				return
			self.conn.sendall(pickle.dumps(message))

class EnvManager(Manager):
	def __init__(self, make_env, client_ports):
		super().__init__(client_ports=client_ports)
		self.num_envs = len(client_ports)
		self.env = make_env()
		self.state_size = [self.env.observation_space.n] if hasattr(self.env.observation_space, "n") else self.env.observation_space.shape
		self.action_size = [self.env.action_space.n] if hasattr(self.env.action_space, "n") else self.env.action_space.shape

	def reset(self):
		self.send_params([pickle.dumps({"cmd": "RESET", "item": [0.0]}) for _ in range(self.num_envs)], encoded=True)
		states = self.await_results(converter=pickle.loads, decoded=True)
		return np.stack(states)

	def step(self, actions, render=False):
		self.send_params([pickle.dumps({"cmd": "STEP", "item": action, "render": render}) for action in actions], encoded=True)
		results = self.await_results(converter=pickle.loads, decoded=True)
		obs, rews, dones, infos = zip(*results)
		return np.stack(obs), np.stack(rews), np.stack(dones), infos

	def close(self):
		self.env.close()
		self.send_params([pickle.dumps({"cmd": "CLOSE", "item": [0.0]}) for _ in range(self.num_envs)], encoded=True)

	def __del__(self):
		self.close()
