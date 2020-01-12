import math
import torch
import random
import numpy as np
from collections import deque
from operator import itemgetter

class BrownianNoise:
	def __init__(self, size, dt=0.02):
		self.size = size
		self.dt = dt
		self.reset()

	def reset(self):
		self.action = np.clip(np.random.randn(1, *self.size), -1, 1)
		self.daction_dt = np.random.randn(1, *self.size)

	def sample(self, state=None, scale=1):
		batch = state.shape[0] if state is not None and len(state.shape) in [2,4] else 1
		self.daction_dt = np.random.randn(batch, *self.size)
		self.action = self.action[0] if len(self.action) != batch else self.action
		self.action = np.clip(self.action + math.sqrt(self.dt) * self.daction_dt, -1, 1)
		return self.action * scale

class RandomAgent():
	def __init__(self, action_size):
		self.noise_process = BrownianNoise(action_size)

	def get_action(self, state, eps=None, sample=True):
		action = self.noise_process.sample(state)
		return action

	def get_env_action(self, env, state=None, eps=None, sample=True):
		action = self.get_action(state, eps, sample)
		action_normal = (1+action)/2
		action_range = env.action_space.high - env.action_space.low
		env_action = env.action_space.low + np.multiply(action_normal, action_range)
		return env_action, action

	def train(self, state, action, next_state, reward, done):
		if done: self.noise_process.reset()

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

class NoisyLinear(torch.nn.Linear):
	def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
		super().__init__(in_features, out_features, bias=bias)
		self.sigma_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features).fill_(sigma_init))
		self.register_buffer("epsilon_weight", torch.zeros(out_features, in_features))
		if bias:
			self.sigma_bias = torch.nn.Parameter(torch.Tensor(out_features).fill_(sigma_init))
			self.register_buffer("epsilon_bias", torch.zeros(out_features))
		self.reset_parameters()

	def reset_parameters(self):
		std = math.sqrt(3 / self.in_features)
		torch.nn.init.uniform_(self.weight, -std, std)
		torch.nn.init.uniform_(self.bias, -std, std)

	def forward(self, input):
		torch.randn(self.epsilon_weight.size(), out=self.epsilon_weight)
		bias = self.bias
		if bias is not None:
			torch.randn(self.epsilon_bias.size(), out=self.epsilon_bias)
			bias = bias + self.sigma_bias * torch.autograd.Variable(self.epsilon_bias)
		weight = self.weight + self.sigma_weight * torch.autograd.Variable(self.epsilon_weight)
		return torch.nn.functional.linear(input, weight, bias)