import gym
import math
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
# from models.rand import ReplayBuffer

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

class ReplayBuffer:
	def __init__(self, capacity):
		self.capacity = capacity
		self.buffer = []
		self.position = 0
	
	def push(self, state, action, reward, next_state, done):
		if len(self.buffer) < self.capacity:
			self.buffer.append(None)
		self.buffer[self.position] = (state, action, reward, next_state, done)
		self.position = (self.position + 1) % self.capacity
	
	def sample(self, batch_size):
		batch = random.sample(self.buffer, batch_size)
		state, action, reward, next_state, done = map(np.stack, zip(*batch))
		return state, action, reward, next_state, done
	
	def __len__(self):
		return len(self.buffer)

class ValueNetwork(torch.nn.Module):
	def __init__(self, state_dim, hidden_dim, init_w=3e-3):
		super(ValueNetwork, self).__init__()
		self.linear1 = torch.nn.Linear(state_dim, hidden_dim)
		self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
		self.linear3 = torch.nn.Linear(hidden_dim, 1)
		# self.linear3.weight.data.uniform_(-init_w, init_w)
		# self.linear3.bias.data.uniform_(-init_w, init_w)
		
	def forward(self, state):
		x = torch.relu(self.linear1(state))
		x = torch.relu(self.linear2(x))
		x = self.linear3(x)
		return x
		
class SoftQNetwork(torch.nn.Module):
	def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3):
		super(SoftQNetwork, self).__init__()
		self.linear1 = torch.nn.Linear(num_inputs + num_actions, hidden_size)
		self.linear2 = torch.nn.Linear(hidden_size, hidden_size)
		self.linear3 = torch.nn.Linear(hidden_size, 1)
		# self.linear3.weight.data.uniform_(-init_w, init_w)
		# self.linear3.bias.data.uniform_(-init_w, init_w)
		
	def forward(self, state, action):
		x = torch.cat([state, action], 1)
		x = torch.relu(self.linear1(x))
		x = torch.relu(self.linear2(x))
		x = self.linear3(x)
		return x
		
class PolicyNetwork(torch.nn.Module):
	def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3, log_std_min=-20, log_std_max=2):
		super(PolicyNetwork, self).__init__()
		self.log_std_min = log_std_min
		self.log_std_max = log_std_max
		self.linear1 = torch.nn.Linear(num_inputs, hidden_size)
		self.linear2 = torch.nn.Linear(hidden_size, hidden_size)
		self.mean_linear = torch.nn.Linear(hidden_size, num_actions)
		# self.mean_linear.weight.data.uniform_(-init_w, init_w)
		# self.mean_linear.bias.data.uniform_(-init_w, init_w)
		self.log_std_linear = torch.nn.Linear(hidden_size, num_actions)
		# self.log_std_linear.weight.data.uniform_(-init_w, init_w)
		# self.log_std_linear.bias.data.uniform_(-init_w, init_w)
		
	def forward(self, state):
		x = torch.relu(self.linear1(state))
		x = torch.relu(self.linear2(x))
		mean = self.mean_linear(x)
		log_std = self.log_std_linear(x)
		# log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
		return mean, log_std
	
	def evaluate(self, state, epsilon=1e-6):
		mean, log_std = self.forward(state)
		std = log_std.exp()
		normal = torch.distributions.Normal(mean, std)
		z = normal.sample()
		action = torch.tanh(z)
		log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + epsilon)
		log_prob = log_prob.sum(-1, keepdim=True)
		return action, log_prob, z, mean, log_std

	def evaluate2(self, state, epsilon=1e-6):
		mean, log_std = self.forward(state)
		std = log_std.exp()
		normal = torch.distributions.Normal(mean, std)
		z = mean + std*torch.distributions.Normal(0, 1).sample(mean.size()).to(device)
		action = torch.tanh(z)
		log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + epsilon)
		log_prob = log_prob.sum(-1, keepdim=True)
		return action, log_prob, z, mean, log_std
		
	def get_action(self, state):
		state = torch.FloatTensor(state).unsqueeze(0).to(device)
		mean, log_std = self.forward(state)
		std = log_std.exp()
		normal = torch.distributions.Normal(mean, std)
		z = normal.sample()
		action = torch.tanh(z)
		action = action.detach().cpu().numpy()
		return action[0]

env = gym.make("Pendulum-v0")
action_dim = env.action_space.shape[0]
state_dim  = env.observation_space.shape[0]
hidden_dim = 256

value_net = ValueNetwork(state_dim, hidden_dim).to(device)
target_value_net = ValueNetwork(state_dim, hidden_dim).to(device)
soft_q_net = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)

for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
	target_param.data.copy_(param.data)
	
value_criterion = torch.nn.MSELoss()
soft_q_criterion = torch.nn.MSELoss()

lr = 3e-4
value_optimizer  = torch.optim.Adam(value_net.parameters(), lr=lr)
soft_q_optimizer = torch.optim.Adam(soft_q_net.parameters(), lr=lr)
policy_optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)

replay_buffer_size = 1000000
replay_buffer = ReplayBuffer(replay_buffer_size)

def soft_q_update(batch_size, gamma=0.99, mean_lambda=1e-3, std_lambda=1e-3, z_lambda=0.0, soft_tau=1e-2):
	state, action, reward, next_state, done = replay_buffer.sample(batch_size)
	state = torch.FloatTensor(state).to(device)
	next_state = torch.FloatTensor(next_state).to(device)
	action = torch.FloatTensor(action).to(device)
	reward = torch.FloatTensor(reward).unsqueeze(1).to(device)
	done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

	expected_q_value = soft_q_net(state, action)
	expected_value   = value_net(state)
	new_action, log_prob, z, mean, log_std = policy_net.evaluate(state)

	target_value = target_value_net(next_state)
	next_q_value = reward + (1 - done) * gamma * target_value
	q_value_loss = soft_q_criterion(expected_q_value, next_q_value.detach())

	expected_new_q_value = soft_q_net(state, new_action)
	next_value = expected_new_q_value - log_prob
	value_loss = value_criterion(expected_value, next_value.detach())

	log_prob_target = expected_new_q_value - expected_value
	policy_loss = (log_prob * (log_prob - log_prob_target).detach()).mean()
	
	mean_loss = mean_lambda * mean.pow(2).mean()
	std_loss = std_lambda * log_std.pow(2).mean()
	z_loss = z_lambda * z.pow(2).sum(1).mean()

	policy_loss += mean_loss + std_loss + z_loss

	soft_q_optimizer.zero_grad()
	q_value_loss.backward()
	soft_q_optimizer.step()

	value_optimizer.zero_grad()
	value_loss.backward()
	value_optimizer.step()

	policy_optimizer.zero_grad()
	policy_loss.backward()
	policy_optimizer.step()
	
	for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
		target_param.data.copy_(target_param.data * (1.0 - soft_tau) + param.data * soft_tau)

def env_action(env, action):
	low  = env.action_space.low
	high = env.action_space.high
	action = low + (action + 1.0) * 0.5 * (high - low)
	action = np.clip(action, low, high)
	return action

def train(self):
	max_frames  = 40000
	max_steps   = 500
	frame_idx   = 0
	rewards     = []
	batch_size  = 128
	while frame_idx < max_frames:
		state = env.reset()
		episode_reward = 0
		for step in range(max_steps):
			action = policy_net.get_action(state)
			action = env_action(env, action)
			next_state, reward, done, _ = env.step(action)
			replay_buffer.push(state, action, reward, next_state, done)
			state = next_state
			episode_reward += reward
			frame_idx += 1
			if len(replay_buffer) > batch_size:
				soft_q_update(batch_size)
			if done:
				break
		print(f"Ep: {frame_idx}, Reward: {episode_reward}")
		rewards.append(episode_reward)
"""
# Advice about PhD in RL

strong academic background
interest in research

after masters have publication
letters from ppl talk about reserach 

ai level
grad students are part of review applications

theory in ML - but got B is red flag

can collaborate with other departments 

recommendations - highest value_loss
ppl who is known that can say good things

cool idea for making a robot do this 

oxtord, OT, princton, Udub

AI residency - apply to 
email peopel and send resume
seminar series related to RL

# Offering CS332

middle of spring

# When to reach out about CA ing

reach out in september
"""