import gym
import torch
import pickle
import numpy as np
from collections import deque
from torchvision import transforms
from models.vae import VAE, LATENT_SIZE
from models.mdrnn import MDRNNCell, HIDDEN_SIZE
from utils.multiprocess import Manager, Worker
from utils.misc import IMG_DIM

FRAME_STACK = 2					# The number of consecutive image states to combine for training a3c on raw images
NUM_ENVS = 16					# The default number of environments to simultaneously train the a3c in parallel

class WorldModel():
	def __init__(self, action_size, num_envs=1, load="", gpu=True):
		self.vae = VAE(load=load, gpu=gpu)
		self.mdrnn = MDRNNCell(load=load, gpu=gpu)
		self.transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((IMG_DIM, IMG_DIM)), transforms.ToTensor()])
		self.state_size = [LATENT_SIZE + HIDDEN_SIZE]
		self.hiddens = {}
		self.reset(num_envs)
		if load: self.load_model(load)

	def reset(self, num_envs, restore=False):
		self.num_envs = num_envs
		self.hidden = self.hiddens[num_envs] if restore and num_envs in self.hiddens else self.mdrnn.init_hidden(num_envs)
		self.hiddens[num_envs] = self.hidden

	def get_state(self, state, numpy=True):
		state = torch.cat([self.transform(s).unsqueeze(0) for s in state]) if self.num_envs > 1 else self.transform(state).unsqueeze(0)
		latent = self.vae.get_latents(state)
		lat_hid = torch.cat([latent, self.hidden[0]], dim=1)
		return lat_hid.cpu().numpy() if numpy else lat_hid, latent

	def step(self, latent, env_action):
		self.hidden = self.mdrnn(env_action.astype(np.float32), latent, self.hidden)

	def load_model(self, dirname="pytorch", name="best"):
		self.vae.load_model(dirname, name)
		self.mdrnn.load_model(dirname, name)
		return self

class ImgStack():
	def __init__(self, action_size, num_envs=1, stack_len=FRAME_STACK, load="", gpu=True):
		self.transform = transforms.Compose([transforms.ToPILImage(), transforms.Grayscale(), transforms.Resize((IMG_DIM, IMG_DIM)), transforms.ToTensor()])
		self.process = lambda x: self.transform(x.astype(np.uint8)).unsqueeze(0).numpy()
		self.state_size = [IMG_DIM, IMG_DIM, stack_len]
		self.stack_len = stack_len
		self.reset(num_envs)

	def reset(self, num_envs, restore=False):
		self.num_envs = num_envs
		self.stack = deque(maxlen=self.stack_len)

	def get_state(self, state):
		state = np.concatenate([self.process(s) for s in state]) if self.num_envs > 1 else self.process(state)
		while len(self.stack) < self.stack_len: self.stack.append(state)
		self.stack.append(state)
		return np.concatenate(self.stack, axis=1), None

	def step(self, state, env_action):
		pass

	def load_model(self, dirname="pytorch", name="best"):
		return self

class StackEnv():
	def __init__(self, env_name, load_dir="pytorch", stack_len=FRAME_STACK, img=False):
		self.env = gym.make(env_name)
		self.env.env.verbose = 0
		self.stack_len = stack_len
		self.img = img
		self.vae = VAE(load=load_dir)
		self.mdrnn = MDRNNCell(load=load_dir)
		self.transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((IMG_DIM, IMG_DIM)), transforms.ToTensor()])
		self.state_size = [IMG_DIM, IMG_DIM, stack_len] if img else [stack_len * LATENT_SIZE + HIDDEN_SIZE * int(stack_len==1)]
		self.action_size = self.env.action_space.shape

	def reset(self):
		state_rgb = self.env.reset()
		state_rgb = rgb2gray(state_rgb) if self.img else state_rgb
		self.hidden = self.mdrnn.init_hidden()
		self.state, self.lat_hid = self.process_state(state_rgb)
		self.stack = [self.state] * self.stack_len
		state = torch.cat(self.stack, dim=1) if self.img or self.stack_len > 1 else self.lat_hid
		return state.squeeze().cpu().detach().numpy()

	def step(self, action, render=False):
		state_rgb, reward, done, info = self.env.step(action)
		state_rgb = rgb2gray(state_rgb) if self.img else state_rgb
		self.hidden = self.update_hidden(np.expand_dims(action.astype(np.float32), axis=0))
		self.state, self.lat_hid = self.process_state(state_rgb)
		self.stack.pop(0)
		self.stack.append(self.state)
		state = torch.cat(self.stack, dim=1) if self.img or self.stack_len > 1 else self.lat_hid
		if render: self.env.render()
		return state.squeeze().cpu().detach().numpy(), reward, done, info

	def process_state(self, state):
		state = self.transform(state).unsqueeze(0)
		if self.img: return state, None
		with torch.no_grad():
			latent = self.vae.get_latents(state)
			lat_hid = torch.cat([latent, self.hidden[0]], dim=1) if self.stack_len <= 1 else None
		return latent, lat_hid

	def update_hidden(self, action):
		hidden = self.hidden
		if not self.img and self.stack_len == 1:
			with torch.no_grad(): hidden = self.mdrnn(action, self.state, self.hidden)
		return hidden

	def close(self):
		self.env.close()

class EnsembleEnv():
	def __init__(self, env_name, num_envs=NUM_ENVS):
		self.env = gym.make(env_name)
		self.env.env.verbose = 0
		self.envs = [gym.make(env_name) for _ in range(num_envs)]
		self.state_size = self.envs[0].observation_space.shape
		self.action_size = self.envs[0].action_space.shape
		for env in self.envs: env.env.verbose = 0

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

class EnvWorker(Worker):
	def __init__(self, self_port, env_name):
		super().__init__(self_port)
		self.env = gym.make(env_name)
		self.env.env.verbose = 0

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
	def __init__(self, env_name, client_ports):
		super().__init__(client_ports=client_ports)
		self.num_envs = len(client_ports)
		self.env = gym.make(env_name)
		self.env.env.verbose = 0
		self.state_size = self.env.observation_space.shape
		self.action_size = self.env.action_space.shape

	def reset(self):
		self.send_params([pickle.dumps({"cmd": "RESET", "item": [0.0]}) for _ in range(self.num_envs)], encoded=True)
		states = self.await_results(converter=pickle.loads, decoded=True)
		return states

	def step(self, actions, render=False):
		self.send_params([pickle.dumps({"cmd": "STEP", "item": action, "render": render}) for action in actions], encoded=True)
		results = self.await_results(converter=pickle.loads, decoded=True)
		states, rewards, dones, infos = map(np.stack, zip(*results))
		return states, rewards, dones, infos

	def close(self):
		self.env.close()
		self.send_params([pickle.dumps({"cmd": "CLOSE", "item": [0.0]}) for _ in range(self.num_envs)], encoded=True)
