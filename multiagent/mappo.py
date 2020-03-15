import torch
import numpy as np
from models.ppo import PPONetwork, PPOCritic
from models.rand import MultiagentReplayBuffer
from utils.network import PTNetwork, PTACNetwork, PTACAgent, INPUT_LAYER, ACTOR_HIDDEN, CRITIC_HIDDEN, LEARN_RATE, NUM_STEPS, MultiheadAttention, one_hot_from_indices

PPO_EPOCHS = 4					# Number of iterations to sample batches for training
BATCH_SIZE = 64					# Number of samples to train on for each train step
EPISODE_BUFFER = 64  	    	# Sets the maximum length of the replay buffer
CLIP_PARAM = 0.2				# The limit of the ratio of new action probabilities to old probabilities
TIME_BATCHES = 10				# The number of batches of time steps to train critic in reverse time sequence
ENTROPY_WEIGHT = 0.001			# The weight for the entropy term of the Actor loss

class MAPPOActor(torch.nn.Module):
	def __init__(self, state_size, action_size):
		super().__init__()
		self.layer1 = torch.nn.Linear(state_size[-1], INPUT_LAYER)
		self.layer2 = torch.nn.Linear(INPUT_LAYER, ACTOR_HIDDEN)
		self.action_probs = torch.nn.Linear(ACTOR_HIDDEN, action_size[-1])
		self.apply(lambda m: torch.nn.init.xavier_normal_(m.weight) if type(m) in [torch.nn.Conv2d, torch.nn.Linear] else None)

	def forward(self, state, action=None, sample=True):
		state = self.layer1(state).relu()
		state = self.layer2(state).relu()
		action_probs = self.action_probs(state).softmax(-1)
		dist = torch.distributions.Categorical(action_probs)
		action_in = dist.sample() if action is None else action.argmax(-1)
		action = one_hot_from_indices(action_in, action_probs.size(-1))
		log_prob = dist.log_prob(action_in)
		entropy = dist.entropy()
		return action, log_prob, entropy

class MAPPONetwork(PTNetwork):
	def __init__(self, state_size, action_size, lr=LEARN_RATE, gpu=True, load=None):
		super().__init__(gpu=gpu, name="mappo")
		self.critic = PPOCritic([np.sum([np.prod(s) for s in state_size])], [np.sum([np.prod(a) for a in action_size])])
		self.models = [PPONetwork(s_size, a_size, MAPPOActor, lambda s,a: self.critic, lr=lr, gpu=gpu, load=load) for s_size,a_size in zip(state_size, action_size)]
		if load: self.load_model(load)

	def get_action_probs(self, state, action_in=None, grad=False, numpy=False, sample=True):
		with torch.enable_grad() if grad else torch.no_grad():
			action_in = [None] * len(state) if action_in is None else action_in
			action_or_entropy, log_prob = map(list, zip(*[model.get_action_probs(s, a, grad=grad, numpy=numpy, sample=sample) for s,a,model in zip(state, action_in, self.models)]))
			return action_or_entropy, log_prob

	def optimize(self, states, actions, states_joint, old_log_probs, rewards, dones, clip_param=CLIP_PARAM, e_weight=ENTROPY_WEIGHT):
		critic_losses = []
		agent = self.models[0]
		next_value = agent.get_value(states_joint)
		next_value = torch.cat([next_value, torch.zeros_like(next_value[:,-1]).unsqueeze(1)], dim=1)
		targets = PTACAgent.compute_ma_gae(rewards[0].unsqueeze(-1), dones[0].unsqueeze(-1), next_value)
		values = torch.zeros_like(targets)
		t_batch = max(rewards[0].size(1)//TIME_BATCHES, 1)
		for t in reversed(range(0,min(rewards[0].size(1), t_batch*TIME_BATCHES),t_batch)):
			values[:,t:t+t_batch] = agent.get_value(states_joint[:,t:t+t_batch], grad=True, numpy=False)
			critic_loss = (values[:,t:t+t_batch] - targets[:,t:t+t_batch].detach()).pow(2).mean()
			critic_losses.append(critic_loss.detach().cpu().numpy())
			agent.step(agent.critic_optimizer, critic_loss, agent.critic_local.parameters(), retain=t>0)	
		advantage = (targets - values).detach()
		
		actor_losses = []
		for model, state, action, old_log_prob in zip(self.models, states, actions, old_log_probs):		
			entropy, new_log_prob = model.get_action_probs(state, action, grad=True, numpy=False)
			ratio = (new_log_prob - old_log_prob).exp()
			ratio_clipped = torch.clamp(ratio, 1.0-clip_param, 1.0+clip_param)
			advantage = advantage.view(*advantage.shape, *[1]*(len(ratio.shape)-len(advantage.shape)))
			actor_loss = -(torch.min(ratio*advantage, ratio_clipped*advantage) + e_weight*entropy).mean()
			model.step(model.actor_optimizer, actor_loss, model.actor_local.parameters())
			actor_losses.append([x.detach().cpu().numpy() for x in [actor_loss, entropy]])
		return [np.mean(critic_losses), *np.mean(actor_losses, axis=0)]

	def save_model(self, dirname="pytorch", name="checkpoint"):
		[PTACNetwork.save_model(model, dirname, f"{name}_{i}", self.name) for i,model in enumerate(self.models)]
		
	def load_model(self, dirname="pytorch", name="checkpoint"):
		[PTACNetwork.load_model(model, dirname, f"{name}_{i}", self.name) for i,model in enumerate(self.models)]

class MAPPOAgent(PTACAgent):
	def __init__(self, state_size, action_size, lr=LEARN_RATE, update_freq=NUM_STEPS, gpu=True, load=None):
		super().__init__(state_size, action_size, MAPPONetwork, lr=lr, update_freq=update_freq, gpu=gpu, load=load)
		self.replay_buffer = MultiagentReplayBuffer(EPISODE_BUFFER)
		self.stats = []

	def get_action(self, state, eps=None, sample=True, numpy=True):
		action, self.log_prob = self.network.get_action_probs(self.to_tensor(state), numpy=True, sample=sample)
		return action

	def train(self, state, action, next_state, reward, done):
		self.buffer.append((state, action, self.log_prob, reward, done))
		if np.any(done[0]):
			states, actions, log_probs, rewards, dones = map(self.to_tensor, zip(*self.buffer))
			states_joint = torch.cat([s.view(*s.size()[:-len(s_size)], np.prod(s_size)) for s,s_size in zip(states, self.state_size)], dim=-1)
			self.replay_buffer.add([self.to_numpy([t.transpose(0,1) for t in x]) for x in (states, actions, [states_joint], log_probs, rewards, dones)])
			self.buffer.clear()
		if len(self.replay_buffer) >= self.replay_buffer.max_steps:
			for _ in range((len(self.replay_buffer)*PPO_EPOCHS)//BATCH_SIZE):
				states, actions, states_joint, log_probs, rewards, dones = self.replay_buffer.sample(BATCH_SIZE, lambda x: torch.Tensor(x).to(self.network.device))
				self.stats.append(self.network.optimize(states, actions, states_joint[0], log_probs, rewards, dones, e_weight=self.eps*ENTROPY_WEIGHT))
			self.replay_buffer.clear()

	def get_stats(self):
		stats = {k:v for k,v in zip(["critic_loss", "actor_loss", "entropy"], np.mean(self.stats, axis=0))} if len(self.stats)>0 else {}
		self.stats = []
		return {**stats, **super().get_stats()}
