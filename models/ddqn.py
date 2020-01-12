import os
import math
import torch
import random
import numpy as np
from models.rand import RandomAgent, PrioritizedReplayBuffer, ReplayBuffer
from utils.network import PTACNetwork, PTACAgent, Conv, INPUT_LAYER, ACTOR_HIDDEN, CRITIC_HIDDEN, LEARN_RATE, NUM_STEPS

EPS_MIN = 0.020               	# The lower limit proportion of random to greedy actions to take
EPS_DECAY = 0.95             	# The rate at which eps decays from EPS_MAX to EPS_MIN
REPLAY_BATCH_SIZE = 32        	# How many experience tuples to sample from the buffer for each train step

class DDQNetwork(PTACNetwork):
	def __init__(self, state_size, action_size, lr=LEARN_RATE, gpu=True, load=None): 
		super().__init__(state_size, action_size, lr=lr, gpu=gpu, load=load)

	def get_action(self, state, use_target=False, numpy=True, sample=True):
		with torch.no_grad():
			q_values = self.critic_local(state) if not use_target else self.critic_target(state)
			return q_values.cpu().numpy() if numpy else q_values

	def get_q_value(self, state, action, use_target=False, numpy=True):
		with torch.no_grad():
			q_values = self.critic_local(state) if not use_target else self.critic_target(state)
			q_values = q_values.reshape(-1, q_values.size(-1))
			q_indices = action.argmax(-1).reshape(-1)
			q_selected = q_values[np.arange(q_indices.size(0)), q_indices].reshape(*state.size()[:-1], 1)
			return q_selected.cpu().numpy() if numpy else q_selected
	
	def optimize(self, states, actions, q_targets, importances=1):
		q_values = self.critic_local(states)[np.arange(actions.size(0)), actions.argmax(-1)].unsqueeze(-1)
		critic_error = q_values - q_targets.detach()
		critic_loss = importances.to(self.device) * critic_error.pow(2)
		self.step(self.critic_optimizer, critic_loss.mean())
		self.soft_copy(self.critic_local, self.critic_target)
		return critic_error.cpu().detach().numpy().squeeze(-1)
	
	def save_model(self, dirname="pytorch", name="best"):
		super().save_model("ddqn", dirname, name)
		
	def load_model(self, dirname="pytorch", name="best"):
		super().load_model("ddqn", dirname, name)

class DDQNAgent(PTACAgent):
	def __init__(self, state_size, action_size, decay=EPS_DECAY, lr=LEARN_RATE, update_freq=NUM_STEPS, gpu=True, load=None):
		super().__init__(state_size, action_size, DDQNetwork, lr=lr, update_freq=update_freq, decay=decay, gpu=gpu, load=load)

	def get_action(self, state, eps=None, sample=True, e_greedy=True):
		eps = self.eps if eps is None else eps
		action_random = super().get_action(state, eps)
		if e_greedy and random.random() < eps: return action_random
		action_greedy = self.network.get_action(self.to_tensor(state), sample=sample)
		return action_greedy
		
	def train(self, state, action, next_state, reward, done):
		self.buffer.append((state, action, reward, done))
		if len(self.buffer) >= int(self.update_freq * (1 - self.eps + EPS_MIN)**0.5):
			states, actions, rewards, dones = map(self.to_tensor, zip(*self.buffer))
			self.buffer.clear()	
			next_state = self.to_tensor(next_state)
			next_action = self.network.get_action(next_state, use_target=False, numpy=False)
			values = self.network.get_q_value(states, actions, use_target=True, numpy=False)
			next_value = self.network.get_q_value(next_state, next_action, use_target=True, numpy=False)
			targets, _ = self.compute_gae(next_value, rewards.unsqueeze(-1), dones.unsqueeze(-1), values)
			states, actions, targets = [x.view(x.size(0)*x.size(1), *x.size()[2:]).cpu().numpy() for x in (states, actions, targets)]
			self.replay_buffer.extend(list(zip(states, actions, targets)), shuffle=True)	
		if len(self.replay_buffer) > 0:
			(states, actions, targets), indices, importances = self.replay_buffer.sample(REPLAY_BATCH_SIZE, dtype=self.to_tensor)
			errors = self.network.optimize(states, actions, targets, importances**(1-self.eps))
			self.replay_buffer.update_priorities(indices, errors)
			if done[0]: self.eps = max(self.eps * self.decay, EPS_MIN)
