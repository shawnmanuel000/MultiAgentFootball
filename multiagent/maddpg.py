import torch
import numpy as np
from models.rand import MultiagentReplayBuffer
from models.ddpg import DDPGCritic, DDPGNetwork
from utils.network import PTNetwork, PTACNetwork, PTACAgent, LEARN_RATE, DISCOUNT_RATE, EPS_MIN, EPS_DECAY, INPUT_LAYER, ACTOR_HIDDEN, CRITIC_HIDDEN, TARGET_UPDATE_RATE, gsoftmax, one_hot

EPS_DECAY = 0.99             	# The rate at which eps decays from EPS_MAX to EPS_MIN
LEARN_RATE = 0.0001           	# Sets how much we want to update the network weights at each training step
ENTROPY_WEIGHT = 0.001			# The weight for the entropy term of the Actor loss
REPLAY_BATCH_SIZE = 38400		# How many experience tuples to sample from the buffer for each train step
MAX_BUFFER_SIZE = 786000		# Sets the maximum length of the replay buffer
NUM_STEPS = 100					# The number of steps to collect experience in sequence for each GAE calculation

class MADDPGActor(torch.nn.Module):
	def __init__(self, state_size, action_size):
		super().__init__()
		self.layer1 = torch.nn.Linear(state_size[-1], INPUT_LAYER)
		self.layer2 = torch.nn.Linear(INPUT_LAYER, ACTOR_HIDDEN)
		self.action_mu = torch.nn.Linear(ACTOR_HIDDEN, action_size[-1])
		self.action_sig = torch.nn.Linear(ACTOR_HIDDEN, action_size[-1])
		self.apply(lambda m: torch.nn.init.xavier_normal_(m.weight) if type(m) in [torch.nn.Conv2d, torch.nn.Linear] else None)

	def forward(self, state, sample=True):
		state = self.layer1(state).relu()
		state = self.layer2(state).relu()
		action_mu = self.action_mu(state)
		action_sig = self.action_sig(state).exp()
		epsilon = torch.randn_like(action_sig)
		action = action_mu + epsilon.mul(action_sig) if sample else action_mu
		return gsoftmax(action, hard=not sample)

class MADDPGNetwork(PTNetwork):
	def __init__(self, state_size, action_size, lr=LEARN_RATE, tau=TARGET_UPDATE_RATE, gpu=True, load=None):
		super().__init__(tau=tau, gpu=gpu, name="maddpg")
		self.critic = lambda s,a: DDPGCritic([np.sum([np.prod(s) for s in state_size])], [np.sum([np.prod(a) for a in action_size])])
		self.models = [DDPGNetwork(s_size, a_size, MADDPGActor, self.critic, lr=lr, gpu=gpu, load=load) for s_size,a_size in zip(state_size, action_size)]
		self.action_size = action_size
		if load: self.load_model(load)

	def get_action_probs(self, state, use_target=False, grad=False, numpy=False, sample=True):
		with torch.enable_grad() if grad else torch.no_grad():
			action_probs = [model.get_action(s, use_target, grad, numpy=numpy, sample=sample) for s,model in zip(state, self.models)]
			return action_probs

	def optimize(self, states, next_states, states_joint, actions_joint, next_states_joint, rewards, dones, e_weight=ENTROPY_WEIGHT):
		stats = []
		next_actions = self.get_action_probs(next_states, grad=False, numpy=False, sample=False)
		actor_actions = self.get_action_probs(states, grad=True, numpy=False, sample=False)
		next_actions_joint = torch.cat([a.view(*a.size()[:-len(a_size)], np.prod(a_size)) for a,a_size in zip(next_actions, self.action_size)], dim=-1)
		actor_actions_joint = torch.cat([a.view(*a.size()[:-len(a_size)], np.prod(a_size)) for a,a_size in zip(actor_actions, self.action_size)], dim=-1)
		for i, (model, action, reward, done) in enumerate(zip(self.models, actor_actions, rewards, dones)):
			next_value = model.get_q_value(next_states_joint, next_actions_joint, use_target=True, numpy=False)
			q_targets = (reward.unsqueeze(-1) + DISCOUNT_RATE * next_value * (1 - done.unsqueeze(-1)))
			q_values = model.get_q_value(states_joint, actions_joint, grad=True, numpy=False)
			critic_loss = (q_values - q_targets.detach()).pow(2).mean()
			model.step(model.critic_optimizer, critic_loss, model.critic_local.parameters())
			model.soft_copy(model.critic_local, model.critic_target)

			actor_loss = -(model.critic_local(states_joint, actor_actions_joint)-q_values.detach()).mean() + e_weight*action.pow(2).mean()
			model.step(model.actor_optimizer, actor_loss, model.actor_local.parameters(), retain=i<len(self.action_size)-1)
			stats.append([x.detach().cpu().numpy() for x in [critic_loss, actor_loss]])
		return np.mean(stats, axis=-1)

	def save_model(self, dirname="pytorch", name="checkpoint"):
		[PTACNetwork.save_model(model, dirname, f"{name}_{i}", self.name) for i,model in enumerate(self.models)]
		
	def load_model(self, dirname="pytorch", name="checkpoint"):
		[PTACNetwork.load_model(model, dirname, f"{name}_{i}", self.name) for i,model in enumerate(self.models)]

class MADDPGAgent(PTACAgent):
	def __init__(self, state_size, action_size, lr=LEARN_RATE, update_freq=NUM_STEPS, decay=EPS_DECAY, gpu=True, load=None):
		super().__init__(state_size, action_size, MADDPGNetwork, lr=lr, update_freq=update_freq, decay=decay, gpu=gpu, load=load)
		self.replay_buffer = MultiagentReplayBuffer(MAX_BUFFER_SIZE)
		self.stats = []

	def get_action(self, state, eps=None, sample=True, numpy=True):
		eps = self.eps if eps is None else eps
		action_random = super().get_action(state, eps)
		action_greedy = self.network.get_action_probs(self.to_tensor(state), sample=sample, numpy=numpy)
		action = [np.tanh((1-eps)*a_greedy + eps*a_random) for a_greedy, a_random in zip(action_greedy, action_random)]
		return action

	def train(self, state, action, next_state, reward, done):
		self.step = 0 if not hasattr(self, "step") else self.step + 1
		self.buffer.append((state, action, next_state, reward, done))
		if len(self.buffer) >= self.update_freq:
			states, actions, next_states, rewards, dones = map(self.to_tensor, zip(*self.buffer))
			self.buffer.clear()	
			states_joint = torch.cat([s.view(*s.size()[:-len(s_size)], np.prod(s_size)) for s,s_size in zip(states, self.state_size)], dim=-1)
			actions_joint = torch.cat([one_hot(a).view(*a.size()[:-len(a_size)], np.prod(a_size)) for a,a_size in zip(actions, self.action_size)], dim=-1)
			next_states_joint = torch.cat([s.view(*s.size()[:-len(s_size)], np.prod(s_size)) for s,s_size in zip(next_states, self.state_size)], dim=-1)
			self.replay_buffer.add([self.to_numpy([t.view(-1, *t.shape[2:]) for t in x]) for x in (states, next_states, [states_joint], [actions_joint], [next_states_joint], rewards, dones)])
			if len(self.replay_buffer) >= REPLAY_BATCH_SIZE:
				states, next_states, states_joint, actions_joint, next_states_joint, rewards, dones = self.replay_buffer.sample(REPLAY_BATCH_SIZE, lambda x: torch.Tensor(x).to(self.network.device))
				self.stats.append(self.network.optimize(states, next_states, states_joint[0], actions_joint[0], next_states_joint[0], rewards, dones))			
		if np.any(done[0]): self.eps = max(self.eps * self.decay, EPS_MIN)

	def get_stats(self):
		stats = {k:v for k,v in zip(["critic_loss", "actor_loss"], np.mean(self.stats, axis=0))} if len(self.stats)>0 else {}
		self.stats = []
		return {**stats, **super().get_stats()}
