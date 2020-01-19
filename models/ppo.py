import gym
import torch
import pickle
import argparse
import numpy as np
from models.rand import ReplayBuffer, PrioritizedReplayBuffer
from utils.network import PTACNetwork, PTACAgent, Conv, INPUT_LAYER, ACTOR_HIDDEN, CRITIC_HIDDEN, LEARN_RATE, DISCOUNT_RATE, NUM_STEPS, ADVANTAGE_DECAY

BATCH_SIZE = 32					# Number of samples to train on for each train step
PPO_EPOCHS = 2					# Number of iterations to sample batches for training
ENTROPY_WEIGHT = 0.005			# The weight for the entropy term of the Actor loss
CLIP_PARAM = 0.05				# The limit of the ratio of new action probabilities to old probabilities

class PPOActor(torch.nn.Module):
	def __init__(self, state_size, action_size):
		super().__init__()
		self.layer1 = torch.nn.Linear(state_size[-1], INPUT_LAYER) if len(state_size)==1 else Conv(state_size, INPUT_LAYER)
		self.layer2 = torch.nn.Linear(INPUT_LAYER, ACTOR_HIDDEN)
		self.layer3 = torch.nn.Linear(ACTOR_HIDDEN, ACTOR_HIDDEN)
		self.action_mu = torch.nn.Linear(ACTOR_HIDDEN, *action_size)
		self.action_sig = torch.nn.Parameter(torch.zeros(*action_size))
		self.apply(lambda m: torch.nn.init.xavier_normal_(m.weight) if type(m) in [torch.nn.Conv2d, torch.nn.Linear] else None)
		
	def forward(self, state, action=None, sample=True):
		state = self.layer1(state).relu()
		state = self.layer2(state).relu()
		state = self.layer3(state).relu()
		action_mu = self.action_mu(state)
		action_sig = self.action_sig.exp().expand_as(action_mu)
		dist = torch.distributions.Normal(action_mu, action_sig)
		action = dist.sample() if action is None else action
		log_prob = dist.log_prob(action)
		entropy = dist.entropy()
		return action, log_prob, entropy

class PPOCritic(torch.nn.Module):
	def __init__(self, state_size, action_size):
		super().__init__()
		self.layer1 = torch.nn.Linear(state_size[-1], INPUT_LAYER) if len(state_size)==1 else Conv(state_size, INPUT_LAYER)
		self.layer2 = torch.nn.Linear(INPUT_LAYER, CRITIC_HIDDEN)
		self.layer3 = torch.nn.Linear(CRITIC_HIDDEN, CRITIC_HIDDEN)
		self.value = torch.nn.Linear(CRITIC_HIDDEN, 1)
		self.apply(lambda m: torch.nn.init.xavier_normal_(m.weight) if type(m) in [torch.nn.Conv2d, torch.nn.Linear] else None)

	def forward(self, state):
		state = self.layer1(state).relu()
		state = self.layer2(state).relu()
		state = self.layer3(state).relu()
		value = self.value(state)
		return value

class PPONetwork(PTACNetwork):
	def __init__(self, state_size, action_size, lr=LEARN_RATE, gpu=True, load=None):
		super().__init__(state_size, action_size, PPOActor, PPOCritic, lr=lr, gpu=gpu, load=load)

	def get_action_probs(self, state, action_in=None, sample=True, grad=True):
		with torch.enable_grad() if grad else torch.no_grad():
			action, log_prob, entropy = self.actor_local(state.to(self.device), action_in, sample)
			return action if action_in is None else entropy.mean(), log_prob

	def get_value(self, state, grad=True):
		with torch.enable_grad() if grad else torch.no_grad():
			return self.critic_local(state.to(self.device))

	def optimize(self, states, actions, old_log_probs, targets, advantages, importances=torch.scalar_tensor(1), clip_param=CLIP_PARAM, e_weight=ENTROPY_WEIGHT, scale=1):
		values = self.get_value(states)
		critic_error = values - targets
		critic_loss = importances.to(self.device) * critic_error.pow(2) * scale
		self.step(self.critic_optimizer, critic_loss.mean())

		entropy, new_log_probs = self.get_action_probs(states, actions)
		ratio = (new_log_probs - old_log_probs).exp()
		ratio_clipped = torch.clamp(ratio, 1.0-clip_param, 1.0+clip_param)
		actor_loss = -(torch.min(ratio*advantages, ratio_clipped*advantages) + e_weight*entropy) * scale
		self.step(self.actor_optimizer, actor_loss.mean())
		return critic_error.cpu().detach().numpy().squeeze(-1)

	def save_model(self, dirname="pytorch", name="best"):
		super().save_model("ppo", dirname, name)
		
	def load_model(self, dirname="pytorch", name="best"):
		super().load_model("ppo", dirname, name)

class PPOAgent(PTACAgent):
	def __init__(self, state_size, action_size, update_freq=NUM_STEPS, lr=LEARN_RATE, gpu=True, load=None):
		super().__init__(state_size, action_size, PPONetwork, lr=lr, update_freq=update_freq, gpu=gpu, load=load)

	def get_action(self, state, eps=None, sample=True):
		state = self.to_tensor(state)
		self.action, self.log_prob = [x.cpu().numpy() for x in self.network.get_action_probs(state, sample=sample, grad=False)]
		return np.tanh(self.action)

	def train(self, state, action, next_state, reward, done):
		self.buffer.append((state, self.action, self.log_prob, reward, done))
		if done[0] or len(self.buffer) >= self.update_freq:
			states, actions, log_probs, rewards, dones = map(self.to_tensor, zip(*self.buffer))
			self.buffer.clear()
			next_state = self.to_tensor(next_state)
			values = self.network.get_value(states, grad=False)
			next_value = self.network.get_value(next_state, grad=False)
			targets, advantages = self.compute_gae(next_value, rewards.unsqueeze(-1), dones.unsqueeze(-1), values, gamma=DISCOUNT_RATE, lamda=ADVANTAGE_DECAY)
			states, actions, log_probs, targets, advantages = [x.view(x.size(0)*x.size(1), *x.size()[2:]) for x in (states, actions, log_probs, targets, advantages)]
			self.replay_buffer.clear().extend(list(zip(states, actions, log_probs, targets, advantages)), shuffle=True)
			for _ in range((len(self.replay_buffer)*PPO_EPOCHS)//BATCH_SIZE):
				state, action, log_prob, target, advantage = self.replay_buffer.next_batch(BATCH_SIZE, torch.stack)
				self.network.optimize(state, action, log_prob, target, advantage, scale=16*dones.size(0)/len(self.replay_buffer))
