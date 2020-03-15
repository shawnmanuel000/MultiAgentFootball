import torch
import numpy as np
from models.rand import MultiagentReplayBuffer
from utils.network import PTACNetwork, PTACAgent, PTCritic, INPUT_LAYER, ACTOR_HIDDEN, CRITIC_HIDDEN, LEARN_RATE, NUM_STEPS, EPS_MIN, TARGET_UPDATE_RATE, one_hot_from_indices

EPS_MIN = 0.01               	# The lower limit proportion of random to greedy actions to take
EPS_DECAY = 0.99             	# The rate at which eps decays from EPS_MAX to EPS_MIN
REPLAY_BATCH_SIZE = 32			# Number of episodes to train on for each train step
EPISODE_BUFFER = 256			# Sets the maximum length of the replay buffer
TIME_BATCHES = 100				# The number of batches of time steps to train critic in reverse time sequence
NUM_STEPS = 250					# The number of steps to collect experience in sequence for each GAE calculation

class COMAActor(torch.nn.Module):
	def __init__(self, state_size, action_size):
		super().__init__()
		self.layer1 = torch.nn.Linear(state_size[-1], INPUT_LAYER)
		self.layer2 = torch.nn.Linear(INPUT_LAYER, ACTOR_HIDDEN)
		self.action_probs = torch.nn.Linear(ACTOR_HIDDEN, action_size[-1])
		self.apply(lambda m: torch.nn.init.xavier_normal_(m.weight) if type(m) in [torch.nn.Conv2d, torch.nn.Linear] else None)

	def forward(self, state, eps):
		state = self.layer1(state).relu()
		state = self.layer2(state).relu()
		action_probs = self.action_probs(state).softmax(-1)
		action_probs = ((1 - eps) * action_probs + torch.ones_like(action_probs).to(state.device) * eps/action_probs.size(-1))
		action = torch.distributions.Categorical(action_probs).sample().long()
		return one_hot_from_indices(action, action_probs.size(-1)), action_probs

class COMANetwork(PTACNetwork):
	def __init__(self, state_size, action_size, lr=LEARN_RATE, tau=TARGET_UPDATE_RATE, gpu=True, load=""):
		self.actor = COMAActor([state_size[0][-1] + action_size[0][-1] + len(state_size)], action_size[0])
		self.critic = lambda s,a: PTCritic([np.sum([np.prod(s) for s in state_size]) + 2*np.sum([np.prod(a) for a in action_size]) + state_size[0][-1] + len(state_size)], action_size[0])
		super().__init__(state_size, action_size, actor=lambda s,a: self.actor, critic=self.critic, lr=lr, gpu=gpu, load=load, name="coma")

	def get_action_probs(self, inputs, eps, grad=False, numpy=False):
		with torch.enable_grad() if grad else torch.no_grad():
			action, action_probs = self.actor_local(inputs, eps)
			return [x.cpu().numpy() if numpy else x for x in [action, action_probs]]

	def optimize(self, actions, critic_inputs, actor_inputs, rewards, dones, eps):
		critic_losses = []
		q_next_value = self.critic_target(critic_inputs)
		q_next_taken = torch.gather(q_next_value, dim=-1, index=actions.argmax(-1, keepdims=True)).squeeze(-1)
		q_next_taken = torch.cat([q_next_taken, torch.zeros_like(q_next_taken[:,-1]).unsqueeze(1)], dim=1)
		q_target = PTACAgent.compute_ma_gae(rewards.unsqueeze(-1), dones.unsqueeze(-1), q_next_taken)
		q_value = torch.zeros_like(q_next_value)
		t_batch = max(rewards.size(1)//TIME_BATCHES, 1)
		for t in reversed(range(0,min(rewards.size(1), t_batch*TIME_BATCHES),t_batch)):
			q_value[:,t:t+t_batch] = self.critic_local(critic_inputs[:,t:t+t_batch])
			q_taken = torch.gather(q_value[:,t:t+t_batch], dim=-1, index=actions[:,t:t+t_batch].argmax(-1, keepdims=True)).squeeze(-1)
			critic_error = (q_taken - q_target[:,t:t+t_batch].detach())
			critic_loss = critic_error.pow(2).mean()
			critic_losses.append(critic_loss.detach().cpu().numpy())
			self.step(self.critic_optimizer, critic_loss, self.critic_local.parameters(), retain=t>0)
		self.soft_copy(self.critic_local, self.critic_target)

		action_probs = self.get_action_probs(actor_inputs, eps, grad=True)[1]
		q_value = q_value.reshape(-1, action_probs.shape[-1])
		pi = action_probs.view(-1, action_probs.shape[-1])
		baseline = (pi * q_value).sum(-1).detach()
		q_taken = torch.gather(q_value, dim=1, index=actions.argmax(-1).reshape(-1, 1)).squeeze(1)
		pi_taken = torch.gather(pi, dim=1, index=actions.argmax(-1).reshape(-1, 1)).squeeze(1)
		advantages = (q_taken - baseline).detach()
		actor_loss = - (advantages * pi_taken.log()).mean()
		self.step(self.actor_optimizer, actor_loss, self.actor_local.parameters())
		return [np.mean(critic_losses), np.mean(actor_loss.detach().cpu().numpy())]

class COMAAgent(PTACAgent):
	def __init__(self, state_size, action_size, update_freq=NUM_STEPS, lr=LEARN_RATE, decay=EPS_DECAY, gpu=True, load=None):
		super().__init__(state_size, action_size, COMANetwork, lr=lr, update_freq=update_freq, decay=decay, gpu=gpu, load=load)
		self.replay_buffer = MultiagentReplayBuffer(EPISODE_BUFFER)
		self.n_agents = len(action_size)
		self.stats = []

	def get_action(self, state, eps=None, sample=True, numpy=True):
		eps = self.eps if eps is None else eps
		obs = np.concatenate(state, -2)
		if not hasattr(self, "action"): self.action = np.zeros([*obs.shape[:-1], self.action_size[0][-1]])
		agent_ids = np.repeat(np.expand_dims(np.eye(self.n_agents), 0), repeats=obs.shape[0], axis=0)
		inputs = torch.from_numpy(np.concatenate([obs, self.action, agent_ids], -1)).float().to(self.network.device)
		self.action = self.network.get_action_probs(inputs, eps=self.eps, numpy=True)[0]
		return np.split(self.action, len(self.action_size), axis=-2)

	def train(self, state, action, next_state, reward, done):
		self.step = 0 if not hasattr(self, "step") else self.step + 1
		self.buffer.append((state, action, reward, done))
		if np.any(done[0]):
			sample = list(map(lambda x: self.to_tensor(x), zip(*self.buffer)))
			states, actions, rewards, dones = map(lambda x: torch.stack(x,2).transpose(0,1), sample)
			obs, actions = [x.squeeze(-2) for x in [states, actions]]
			state = states.repeat(1,1,1,self.n_agents,1).view(*states.shape[:3],-1)
			actions_joint = actions.view(*actions.shape[:2],1,-1).repeat(1,1,self.n_agents,1)
			agent_mask = (1-torch.eye(self.n_agents, device=self.network.device))
			agent_mask = agent_mask.view(-1, 1).repeat(1, self.action_size[0][-1]).view(self.n_agents, -1).unsqueeze(0).unsqueeze(0)
			last_actions = torch.cat([torch.zeros_like(actions[:, 0:1]), actions[:, :-1]], dim=1)
			last_actions_joint = last_actions.view(*last_actions.shape[:2],1,-1).repeat(1,1,self.n_agents,1)
			agent_inds = torch.eye(self.n_agents, device=self.network.device).unsqueeze(0).unsqueeze(0).expand(*obs.shape[:2],-1,-1)
			critic_inputs = torch.cat([state, obs, actions_joint * agent_mask, last_actions_joint, agent_inds], dim=-1)
			actor_inputs = torch.cat([obs, last_actions, agent_inds], dim=-1)
			self.replay_buffer.add([self.to_numpy([x]) for x in (actions, critic_inputs, actor_inputs, rewards, dones)])
			self.buffer.clear()
		if (self.step % self.update_freq)==0 and len(self.replay_buffer) >= REPLAY_BATCH_SIZE:
			actions, critic_inputs, actor_inputs, rewards, dones = [x[0] for x in self.replay_buffer.sample(REPLAY_BATCH_SIZE, lambda x: torch.Tensor(x).to(self.network.device))]
			self.stats.append(self.network.optimize(actions, critic_inputs, actor_inputs, rewards.mean(-1), dones.mean(-1), self.eps))
		if np.any(done[0]): self.eps = max(self.eps * self.decay, EPS_MIN)

	def get_stats(self):
		stats = {k:v for k,v in zip(["critic_loss", "actor_loss"], np.mean(self.stats, axis=0))} if len(self.stats)>0 else {}
		self.stats = []
		return {**stats, **super().get_stats()}
