import gym
import math
import torch
import random
import numpy as np
import scipy.special as sps
from collections import deque
from operator import itemgetter

class Noise():
	def __init__(self, size):
		self.size = size

	def reset(self):
		pass

	def sample(self, shape=[], scale=1):
		return scale * np.random.randn(*shape, *self.size)

class OUNoise(Noise):
	def __init__(self, size, scale=0.1, mu=0, theta=0.15, sigma=0.2):
		self.size = size
		self.scale = scale
		self.mu = mu
		self.theta = theta
		self.sigma = sigma
		self.state = np.ones(*self.size) * self.mu
		self.reset()

	def reset(self, shape=[]):
		self.state = np.ones(*shape, *self.size) * self.mu

	def sample(self, shape=[], scale=1):
		delta = self.sigma * np.random.randn(*self.state.shape)
		if self.state.shape != delta.shape: self.reset(shape)
		x = self.state
		dx = self.theta * (self.mu - x) + delta
		self.state = x + dx
		return self.state * self.scale

class BrownianNoise(Noise):
	def __init__(self, size, dt=0.2):
		self.size = size
		self.dt = dt
		self.reset()

	def reset(self):
		self.action = np.clip(np.random.randn(*self.size), -1, 1)
		self.daction_dt = np.random.randn(*self.size)

	def sample(self, shape=[], scale=1):
		self.daction_dt = np.random.randn(*shape, *self.size)
		self.action = np.zeros_like(self.daction_dt) if self.action.shape != self.daction_dt.shape else self.action
		self.action = np.clip(self.action + math.sqrt(self.dt) * self.daction_dt, -1, 1)
		return self.action * scale

class RandomAgent():
	def __init__(self, state_size, action_size, eps=1.0, **kwargs):
		self.noise_process = BrownianNoise(action_size) if type(action_size[0]) in [int, np.int32] else [Noise(a_size) for a_size in action_size]
		self.action_size = action_size
		self.state_size = state_size
		self.eps = eps

	def get_action(self, state, eps=None, sample=True):
		if type(self.state_size[0]) in [int, np.int32] and type(self.action_size[0]) in [int, np.int32]:
			return self.noise_process.sample(state.shape[:-len(self.state_size)])
		return [p.sample(s.shape[:-len(s_size)]) for p,s,s_size in zip(self.noise_process, state, self.state_size)]

	def get_env_action(self, env, state=None, eps=None, sample=True):
		action = self.get_action(state, eps, sample)
		env_action = self.to_env_action(env.action_space, action)
		return env_action, action

	@staticmethod
	def to_env_action(action_space, action):
		if type(action_space) == list: return [RandomAgent.to_env_action(a_space, a) for a_space,a in zip(action_space, action)]
		if type(action_space) in [gym.spaces.Discrete, gym.spaces.MultiDiscrete]: return np.argmax(action, -1)
		return action_space.low + np.multiply((1+action)/2, action_space.high - action_space.low)

	def train(self, state, action, next_state, reward, done):
		if np.any(done[0]): self.noise_process.reset() if type(self.noise_process) != list else [p.reset() for p in self.noise_process]

	def get_stats(self):
		return {"eps": self.eps}

class ReplayBuffer():
	def __init__(self, maxlen=None):
		self.buffer = deque(maxlen=maxlen)
		
	def add(self, experience):
		self.buffer.append(experience)
		return self

	def extend(self, experiences, shuffle=False):
		if shuffle: random.shuffle(experiences)
		for exp in experiences:
			self.add(exp)
		return self

	def clear(self):
		self.buffer.clear()
		self.i_batch = 0
		return self
		
	def sample(self, batch_size, dtype=np.array, weights=None):
		sample_size = min(len(self.buffer), batch_size)
		sample_indices = random.choices(range(len(self.buffer)), k=sample_size, weights=weights)
		samples = itemgetter(*sample_indices)(self.buffer)
		sample_arrays = samples if dtype is None else map(dtype, zip(*samples))
		return sample_arrays, sample_indices, torch.Tensor([1])

	def next_batch(self, batch_size=1, dtype=np.array):
		if not hasattr(self, "i_batch"): self.i_batch = 0
		sample_indices = [i%len(self.buffer) for i in range(self.i_batch, self.i_batch+batch_size)]
		samples = itemgetter(*sample_indices)(self.buffer)
		self.i_batch = (self.i_batch+batch_size) % len(self.buffer)
		return map(dtype, zip(*samples))

	def update_priorities(self, indices, errors, offset=0.1):
		pass

	def reset_priorities(self):
		pass

	def __len__(self):
		return len(self.buffer)

class PrioritizedReplayBuffer(ReplayBuffer):
	def __init__(self, maxlen=None):
		super().__init__(maxlen)
		self.priorities = deque(maxlen=maxlen)
		
	def add(self, experience):
		super().add(experience)
		self.priorities.append(max(self.priorities, default=1))
		return self

	def clear(self):
		super().clear()
		self.priorities.clear()
		return self
		
	def get_probabilities(self, priority_scale):
		scaled_priorities = np.array(self.priorities) ** priority_scale
		sample_probabilities = scaled_priorities / sum(scaled_priorities)
		return sample_probabilities
	
	def get_importance(self, probabilities):
		importance = 1/len(self.buffer) * 1/probabilities
		importance_normalized = importance / max(importance)
		return importance_normalized[:,np.newaxis]
		
	def sample(self, batch_size, dtype=np.array, priority_scale=0.5):
		sample_probs = self.get_probabilities(priority_scale)
		samples, sample_indices, _ = super().sample(batch_size, None, sample_probs)
		importance = self.get_importance(sample_probs[sample_indices])
		return map(dtype, zip(*samples)), sample_indices, torch.Tensor(importance)
						
	def update_priorities(self, indices, errors, offset=0.1):
		for i,e in zip(indices, errors):
			self.priorities[i] = abs(e) + offset

	def reset_priorities(self):
		for i in range(len(self.priorities)):
			self.priorities[i] = 1
			
class MultiagentReplayBuffer():
	def __init__(self, max_steps):
		self.max_steps = max_steps
		self.clear()

	def clear(self):
		self.initialized = False
		self.filled_i = 0  # index of first empty location in buffer (last index when full)
		self.curr_i = 0  # current index to write to (ovewrite oldest data)
		return self

	def __len__(self):
		return self.filled_i

	def initialize(self, experience):
		self.buffers = [[np.zeros((self.max_steps, *agent.shape[1:])) for agent in exp] for exp in experience]
		self.initialized = True

	def add(self, experience):
		if not self.initialized: self.initialize(experience)
		nentries = experience[0][0].shape[0]  # handle multiple parallel environments
		if self.curr_i + nentries > self.max_steps:
			rollover = self.max_steps - self.curr_i # num of indices to roll over
			for buffer,exp in zip(self.buffers, experience):
				for buf,agent in zip(buffer, exp):
					buf = np.roll(buf, rollover, axis=0)
			self.curr_i = 0
			self.filled_i = self.max_steps
		for buffer,exp in zip(self.buffers, experience):
			for buf,agent in zip(buffer, exp):
				buf[self.curr_i:self.curr_i + nentries] = agent
		self.curr_i += nentries
		if self.filled_i < self.max_steps: self.filled_i += nentries
		if self.curr_i == self.max_steps: self.curr_i = 0

	def sample(self, N, cast, norm_rews=False):
		N = min(N, self.filled_i)
		inds = np.random.choice(np.arange(self.filled_i), size=N, replace=False)
		return [[cast(buf[inds]) for buf in buffer] for buffer in self.buffers]
