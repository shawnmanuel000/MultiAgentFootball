import os
import math
import torch
import random
import numpy as np
from models.rand import RandomAgent, ReplayBuffer

REG_LAMBDA = 1e-6             	# Penalty multiplier to apply for the size of the network weights
LEARN_RATE = 0.0003           	# Sets how much we want to update the network weights at each training step
TARGET_UPDATE_RATE = 0.001   	# How frequently we want to copy the local network to the target network (for double DQNs)
INPUT_LAYER = 256				# The number of output nodes from the first layer to Actor and Critic networks
ACTOR_HIDDEN = 512				# The number of nodes in the hidden layers of the Actor network
CRITIC_HIDDEN = 1024			# The number of nodes in the hidden layers of the Critic networks

DISCOUNT_RATE = 0.998			# The discount rate to use in the Bellman Equation
NUM_STEPS = 500					# The number of steps to collect experience in sequence for each GAE calculation
EPS_MAX = 1.0                 	# The starting proportion of random to greedy actions to take
EPS_MIN = 0.001               	# The lower limit proportion of random to greedy actions to take
EPS_DECAY = 0.980             	# The rate at which eps decays from EPS_MAX to EPS_MIN
ADVANTAGE_DECAY = 0.95			# The discount factor for the cumulative GAE calculation
MAX_BUFFER_SIZE = 1000000      	# Sets the maximum length of the replay buffer
REPLAY_BATCH_SIZE = 32        	# How many experience tuples to sample from the buffer for each train step

class Conv(torch.nn.Module):
	def __init__(self, state_size, output_size):
		super().__init__()
		self.conv1 = torch.nn.Conv2d(state_size[-1], 32, kernel_size=4, stride=2)
		self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=4, stride=2)
		self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=4, stride=2)
		self.conv4 = torch.nn.Conv2d(128, 256, kernel_size=4, stride=2)
		self.linear1 = torch.nn.Linear(self.get_conv_output(state_size), output_size)
		self.apply(lambda m: torch.nn.init.xavier_normal_(m.weight) if type(m) in [torch.nn.Conv2d, torch.nn.Linear] else None)

	def forward(self, state):
		out_dims = state.size()[:-3]
		state = state.view(-1, *state.size()[-3:])
		state = self.conv1(state).tanh()
		state = self.conv2(state).tanh() 
		state = self.conv3(state).tanh() 
		state = self.conv4(state).tanh() 
		state = state.view(state.size(0), -1)
		state = self.linear1(state).tanh()
		state = state.view(*out_dims, -1)
		return state

	def get_conv_output(self, state_size):
		inputs = torch.randn(1, state_size[-1], *state_size[:-1])
		output = self.conv4(self.conv3(self.conv2(self.conv1(inputs))))
		return np.prod(output.size())

class PTActor(torch.nn.Module):
	def __init__(self, state_size, action_size):
		super().__init__()
		self.state_fc1 = torch.nn.Linear(state_size[-1], INPUT_LAYER) if len(state_size)!=3 else Conv(state_size, INPUT_LAYER)
		self.state_fc2 = torch.nn.Linear(INPUT_LAYER, ACTOR_HIDDEN)
		self.state_fc3 = torch.nn.Linear(ACTOR_HIDDEN, ACTOR_HIDDEN)
		self.action_mu = torch.nn.Linear(ACTOR_HIDDEN, action_size[-1])
		self.apply(lambda m: torch.nn.init.xavier_normal_(m.weight) if type(m) in [torch.nn.Conv2d, torch.nn.Linear] else None)

	def forward(self, state):
		state = self.state_fc1(state).relu() 
		state = self.state_fc2(state).relu() 
		state = self.state_fc3(state).relu() 
		action_mu = self.action_mu(state)
		return action_mu

class PTCritic(torch.nn.Module):
	def __init__(self, state_size, action_size=[1]):
		super().__init__()
		self.state_fc1 = torch.nn.Linear(state_size[-1], INPUT_LAYER) if len(state_size)!=3 else Conv(state_size, INPUT_LAYER)
		self.state_fc2 = torch.nn.Linear(INPUT_LAYER, CRITIC_HIDDEN)
		self.state_fc3 = torch.nn.Linear(CRITIC_HIDDEN, CRITIC_HIDDEN)
		self.value = torch.nn.Linear(CRITIC_HIDDEN, action_size[-1])
		self.apply(lambda m: torch.nn.init.xavier_normal_(m.weight) if type(m) in [torch.nn.Conv2d, torch.nn.Linear] else None)

	def forward(self, state, action=None):
		state = self.state_fc1(state).relu()
		state = self.state_fc2(state).relu()
		state = self.state_fc3(state).relu()
		value = self.value(state)
		return value

class PTNetwork():
	def __init__(self, tau=TARGET_UPDATE_RATE, gpu=True, name="pt"): 
		self.tau = tau
		self.name = name
		self.device = torch.device('cuda' if gpu and torch.cuda.is_available() else 'cpu')

	def init_weights(self, model=None):
		model = self if model is None else model
		model.apply(lambda m: torch.nn.init.xavier_normal_(m.weight) if type(m) in [torch.nn.Conv2d, torch.nn.Linear] else None)
		
	def step(self, optimizer, loss, param_norm=None, retain=False):
		optimizer.zero_grad()
		loss.backward(retain_graph=retain)
		if param_norm is not None: torch.nn.utils.clip_grad_norm_(param_norm, 0.5)
		optimizer.step()

	def soft_copy(self, local, target):
		for t,l in zip(target.parameters(), local.parameters()):
			t.data.copy_(t.data + self.tau*(l.data - t.data))

class PTQNetwork(PTNetwork):
	def __init__(self, state_size, action_size, critic=PTCritic, lr=LEARN_RATE, tau=TARGET_UPDATE_RATE, gpu=True, load="", name="ptq"): 
		super().__init__(tau, gpu, name)
		self.critic_local = critic(state_size, action_size).to(self.device)
		self.critic_target = critic(state_size, action_size).to(self.device)
		self.critic_optimizer = torch.optim.Adam(self.critic_local.parameters(), lr=lr, weight_decay=REG_LAMBDA)
		if load: self.load_model(load)

	def save_model(self, net="qlearning", dirname="pytorch", name="checkpoint"):
		filepath = get_checkpoint_path(net, dirname, name)
		os.makedirs(os.path.dirname(filepath), exist_ok=True)
		torch.save(self.critic_local.state_dict(), filepath)
		
	def load_model(self, net="qlearning", dirname="pytorch", name="checkpoint"):
		filepath = get_checkpoint_path(net, dirname, name)
		if os.path.exists(filepath):
			self.critic_local.load_state_dict(torch.load(filepath, map_location=self.device))
			self.critic_target.load_state_dict(torch.load(filepath, map_location=self.device))

class PTACNetwork(PTNetwork):
	def __init__(self, state_size, action_size, actor=PTActor, critic=PTCritic, lr=LEARN_RATE, tau=TARGET_UPDATE_RATE, gpu=True, load="", name="ptac"): 
		super().__init__(tau, gpu, name)
		self.actor_local = actor(state_size, action_size).to(self.device)
		self.actor_target = actor(state_size, action_size).to(self.device)
		self.actor_optimizer = torch.optim.Adam(self.actor_local.parameters(), lr=lr, weight_decay=REG_LAMBDA)
		
		self.critic_local = critic(state_size, action_size).to(self.device)
		self.critic_target = critic(state_size, action_size).to(self.device)
		self.critic_optimizer = torch.optim.Adam(self.critic_local.parameters(), lr=lr, weight_decay=REG_LAMBDA)
		if load: self.load_model(load)

	def save_model(self, dirname="pytorch", name="checkpoint", net=None):
		filepath = get_checkpoint_path(self.name if net is None else net, dirname, name)
		os.makedirs(os.path.dirname(filepath), exist_ok=True)
		torch.save(self.actor_local.state_dict(), filepath.replace(".pth", "_a.pth"))
		torch.save(self.critic_local.state_dict(), filepath.replace(".pth", "_c.pth"))
		
	def load_model(self, dirname="pytorch", name="checkpoint", net=None):
		filepath = get_checkpoint_path(self.name if net is None else net, dirname, name)
		if os.path.exists(filepath.replace(".pth", "_a.pth")):
			try:
				self.actor_local.load_state_dict(torch.load(filepath.replace(".pth", "_a.pth"), map_location=self.device))
				self.actor_target.load_state_dict(torch.load(filepath.replace(".pth", "_a.pth"), map_location=self.device))
				self.critic_local.load_state_dict(torch.load(filepath.replace(".pth", "_c.pth"), map_location=self.device))
				self.critic_target.load_state_dict(torch.load(filepath.replace(".pth", "_c.pth"), map_location=self.device))
				print(f"Loaded model at {filepath}")
			except:
				pass

class PTACAgent(RandomAgent):
	def __init__(self, state_size, action_size, network=PTACNetwork, lr=LEARN_RATE, update_freq=NUM_STEPS, eps=EPS_MAX, decay=EPS_DECAY, gpu=True, load=None):
		super().__init__(state_size, action_size, eps)
		self.network = network(state_size, action_size, lr=lr, gpu=gpu, load=load)
		self.replay_buffer = ReplayBuffer(MAX_BUFFER_SIZE)
		self.update_freq = update_freq
		self.buffer = []
		self.decay = decay

	def to_tensor(self, arr):
		if isinstance(arr, np.ndarray): return torch.from_numpy(arr.astype(np.float32)).float().to(self.network.device)
		if isinstance(arr, list) and (isinstance(arr[0], list) or isinstance(arr[0], np.ndarray)): return [self.to_tensor(a) for a in arr]
		if isinstance(arr, tuple) and type(arr[0]) in [list, tuple]:
			result = [self.to_tensor(list(a)) for a in arr]
			return [torch.stack(x, dim=0) for x in zip(*result)] if isinstance(result[0], list) else torch.stack(result, dim=0)
		return self.to_tensor(np.array(arr))

	def to_numpy(self, arr):
		if isinstance(arr, torch.Tensor): return arr.cpu().numpy()
		if isinstance(arr, list) and (isinstance(arr[0], list) or isinstance(arr[0], torch.Tensor)): return [self.to_numpy(a) for a in arr]
		return arr

	def get_action(self, state, eps=None, e_greedy=False):
		action_random = super().get_action(state, eps)
		return action_random

	@staticmethod
	def compute_gae(last_value, rewards, dones, values, gamma=DISCOUNT_RATE, lamda=ADVANTAGE_DECAY):
		with torch.no_grad():
			gae = 0
			if dones.size() != rewards.size(): dones = dones.unsqueeze(-1)
			targets = torch.zeros_like(values, device=values.device)
			values = torch.cat([values, last_value.unsqueeze(0)])
			for step in reversed(range(len(rewards))):
				delta = rewards[step] + gamma * values[step + 1] * (1-dones[step]) - values[step]
				gae = delta + gamma * lamda * (1-dones[step]) * gae
				targets[step] = gae + values[step]
			advantages = targets - values[:-1]
			return targets, advantages

	@staticmethod
	def compute_ma_gae(rewards, done, target_qs, gamma=DISCOUNT_RATE, lamda=ADVANTAGE_DECAY):
		ret = target_qs.new_zeros(*target_qs.shape)
		ret[:,-1] = target_qs[:,-1]*(1-torch.sum(done, dim=1))
		for t in range(ret.shape[1]-2, -1, -1):
			ret[:,t] = lamda*gamma*ret[:,t+1] + (rewards[:,t]+(1-lamda)*gamma*target_qs[:,t+1]*(1-done[:,t]))
		return ret[:,0:-1]
		
	def train(self, state, action, next_state, reward, done):
		pass

class IntrinsicCuriosityModule(PTNetwork, torch.nn.Module):
	def __init__(self, state_size, action_size, lr=LEARN_RATE, gpu=True):
		torch.nn.Module.__init__(self)
		PTNetwork.__init__(self, gpu=gpu, name="icm")
		self.state_features = torch.nn.Linear(state_size[-1], ACTOR_HIDDEN).to(self.device)
		self.forward_state_pred = torch.nn.Linear(ACTOR_HIDDEN+action_size[-1], ACTOR_HIDDEN).to(self.device)
		self.inverse_action_pred = torch.nn.Linear(2*ACTOR_HIDDEN, action_size[-1]).to(self.device)
		self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=REG_LAMBDA)
		self.total_rewards = []

	def train(self, state, action, next_state, reward, done, eta=0.1, beta=0.5):
		state, action, next_state = [torch.from_numpy(x).float().to(self.device) for x in [state, action, next_state]]
		state_features = self.state_features(state)
		next_state_features = self.state_features(next_state)
		next_state_pred = self.forward_state_pred(torch.cat([state_features, action], dim=-1))
		action_pred = self.inverse_action_pred(torch.cat([state_features, next_state_features], dim=-1))
		forward_error = (next_state_features - next_state_pred).pow(2).mean(-1)
		inverse_error = (action - action_pred).pow(2).mean(-1)
		self.step(self.optimizer, (forward_error+inverse_error).mean(), self.parameters())

		intrinsic = eta*(beta*forward_error + (1-beta)*inverse_error).view(*reward.shape).cpu().detach().numpy() * (1-done)
		self.total_rewards.append(np.mean(intrinsic))
		return reward + intrinsic

	def save_model(self, dirname="pytorch", name="checkpoint", net=None):
		filepath = get_checkpoint_path(self.name if net is None else net, dirname, name)
		os.makedirs(os.path.dirname(filepath), exist_ok=True)
		torch.save(self.state_dict(), filepath)
		
	def load_model(self, dirname="pytorch", name="checkpoint", net=None):
		filepath = get_checkpoint_path(self.name if net is None else net, dirname, name)
		if os.path.exists(filepath):
			self.load_state_dict(torch.load(filepath, map_location=self.device))

	def get_stats(self):
		stats = np.mean(self.total_rewards)
		self.total_rewards = []
		return stats

class MultiheadAttention(torch.nn.Module):
	def __init__(self, d_model, num_heads, depth):
		super().__init__()
		self.depth = depth
		self.num_heads = num_heads
		self.wq = torch.nn.Linear(1, num_heads*depth)
		self.wk = torch.nn.Linear(1, num_heads*depth)
		self.wv = torch.nn.Linear(1, num_heads*depth)
		self.norm = torch.nn.LayerNorm(d_model)

	def scaled_dot_product_attention(self, q, k, v):
		dk = k.shape[-1]
		kt = k.transpose(-1, -2)
		qk = torch.matmul(q, kt) / np.sqrt(dk)
		qkmax = qk.max(-1, keepdim=True).values
		attn_weights = (qk - qkmax).softmax(-1)
		scaled_attn = torch.matmul(attn_weights, v)
		return scaled_attn, attn_weights

	def forward(self, x):
		# X: (BS, T, NE, f=1)
		x = x.unsqueeze(-1)
		q = self.wq(x).reshape(*x.shape[:-1], self.num_heads, self.depth)
		k = self.wk(x).reshape(*x.shape[:-1], self.num_heads, self.depth)
		v = self.wv(x).reshape(*x.shape[:-1], self.num_heads, self.depth)
		q, k, v = [s.transpose(-2, -3) for s in [q,k,v]]
		scaled_attn, _ = self.scaled_dot_product_attention(q, k, v)
		scaled_attn = scaled_attn.transpose(-2, -3)
		out = scaled_attn.reshape(*scaled_attn.shape[:-2], -1)
		add = (x + out).squeeze(-1)
		norm = self.norm(add)
		return norm

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

def get_checkpoint_path(net="qlearning", dirname="pytorch", name="checkpoint"):
	return os.path.normpath(f"./saved_models/{net}/{dirname}/{name}.pth")

# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def sample_gumbel(shape, eps=1e-20, tens_type=torch.FloatTensor):
    """Sample from Gumbel(0, 1)"""
    U = torch.autograd.Variable(tens_type(*shape).uniform_(), requires_grad=False)
    return -torch.log(-torch.log(U + eps) + eps)

# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def gumbel_softmax_sample(logits, temperature):
	""" Draw a sample from the Gumbel-Softmax distribution"""
	y = logits + sample_gumbel(logits.shape, tens_type=type(logits.data)).to(logits.device)
	return torch.nn.functional.softmax(y / temperature, dim=-1)

# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def gsoftmax(logits, temperature=1.0, hard=True):
	"""Sample from the Gumbel-Softmax distribution and optionally discretize.
	Args:
	logits: [batch_size, n_class] unnormalized log-probs
	temperature: non-negative scalar
	hard: if True, take argmax, but differentiate w.r.t. soft sample y
	Returns:
	[batch_size, n_class] sample from the Gumbel-Softmax distribution.
	If hard=True, then the returned sample will be one-hot, otherwise it will
	be a probabilitiy distribution that sums to 1 across classes
	"""
	y = gumbel_softmax_sample(logits, temperature)
	if hard:
		y_hard = one_hot(y)
		y = (y_hard - y).detach() + y
	return y

def one_hot(logits):
	return (logits == logits.max(-1, keepdim=True)[0]).float().to(logits.device)

def one_hot_from_indices(indices, depth):
	y_onehot = torch.zeros([*indices.shape, depth]).to(indices.device)
	y_onehot.scatter_(-1, indices.unsqueeze(-1).long(), 1)
	return y_onehot.float()
