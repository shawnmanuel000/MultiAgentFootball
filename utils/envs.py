import gym
import pickle
import numpy as np
from mpi4py import MPI
from utils.multiprocess import TCPClient, TCPServer, MPIConnection, MPI_SIZE, MPI_RANK

NUM_ENVS = 16					# The default number of environments to simultaneously train the a3c in parallel
MPI_COMM = MPI.COMM_WORLD
MPI_SIZE = MPI_COMM.Get_size()

def get_space_size(space):
	if isinstance(space, gym.spaces.MultiDiscrete): return [*space.shape, space.nvec[0]]
	if isinstance(space, gym.spaces.Discrete): return [space.n]
	if isinstance(space, gym.spaces.Box): return space.shape
	if isinstance(space, list): return [get_space_size(sp) for sp in space]
	raise ValueError()

def stack(values):
	if isinstance(values[0], list): return [stack(arr) for arr in zip(*values)]
	return np.stack(values)

def unstack(values):
	if isinstance(values, list): return [unstack(arr) for arr in zip(*values)]
	return values

class EnsembleEnv():
	def __init__(self, make_env, num_envs=NUM_ENVS):
		self.num_envs = num_envs
		self.env = make_env()
		self.envs = [make_env() for _ in range(num_envs)]
		self.test_envs = [make_env() for _ in range(num_envs)]
		self.state_size = get_space_size(self.env.observation_space)
		self.action_size = get_space_size(self.env.action_space)
		self.action_space = self.env.action_space

	def reset(self, train=False):
		obs = [env.reset() for env in (self.envs if train else self.test_envs)]
		return stack(obs)

	def step(self, actions, train=False, render=False):
		results = []
		actions = unstack(actions)
		envs = self.envs if train else self.test_envs
		for env,action in zip(envs, actions):
			state, rew, done, info = env.step(action, train)
			state = env.reset() if train and np.all(done) else state
			results.append((state, rew, done, info))
			if render: env.render()
		obs, rews, dones, infos = zip(*results)
		return stack(obs), stack(rews), stack(dones), infos

	def render(self, train=False):
		self.test_envs[0].render()

	def close(self):
		self.env.close()
		for env in self.envs: env.close()
		for env in self.test_envs: env.close()

	def __del__(self):
		self.close()

class EnvManager():
	def __init__(self, make_env, server_ports):
		self.env = make_env()
		self.state_size = get_space_size(self.env.observation_space)
		self.action_size = get_space_size(self.env.action_space)
		self.action_space = self.env.action_space
		self.conn = TCPClient(server_ports) if MPI_SIZE==1 else MPIConnection()
		self.num_envs = len(server_ports) if MPI_SIZE==1 else MPI_SIZE-1

	def reset(self, train=False):
		self.conn.broadcast([{"cmd": "RESET", "item": [0.0], "train": train} for _ in range(self.num_envs)])
		obs = self.conn.gather()
		return stack(obs)

	def step(self, actions, train=False, render=False):
		actions = unstack(actions)
		self.conn.broadcast([{"cmd": "STEP", "item": action, "render": render, "train": train} for action in actions])
		results = self.conn.gather()
		obs, rews, dones, infos = zip(*results)
		return stack(obs), stack(rews), stack(dones), infos

	def render(self, num=1, train=False):
		self.conn.broadcast([{"cmd": "RENDER", "train": train} for _ in range(min(num, self.num_envs))])

	def close(self):
		self.env.close()
		self.conn.broadcast([{"cmd": "CLOSE", "item": [0.0]} for _ in range(self.num_envs)])

	def __del__(self):
		self.close()

class EnvWorker():
	def __init__(self, self_port, make_env):
		self.env = [make_env(), make_env()]
		self.conn = TCPServer(self_port) if MPI_SIZE==1 else MPIConnection()

	def start(self, log=MPI_SIZE==1):
		step = 0
		rewards = [None, None]
		while True:
			data = self.conn.recv()
			train = data.get("train", False)
			env = self.env[int(train)]
			if data["cmd"] == "RESET":
				message = env.reset()
				rewards[int(train)] = None
			elif data["cmd"] == "STEP":
				state, reward, done, info = env.step(data["item"], train)
				state = env.reset() if train and np.all(done) else state
				rewards[int(train)] = np.array(reward) if rewards[int(train)] is None else rewards[int(train)] + np.array(reward)
				message = (state, reward, done, info)
				step += int(train)
				if log and train and np.all(done): 
					print(f"Step: {step}, Reward: {rewards[int(train)]}")
					rewards[int(train)] = None
			elif data["cmd"] == "RENDER":
				env.render()
				continue
			elif data["cmd"] == "CLOSE":
				[env.close() for env in self.env]
				return
			self.conn.send(message)
