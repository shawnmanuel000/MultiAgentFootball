import os
import re
import gym
import random
import numpy as np
from collections import deque
from utils.misc import rgb2gray, resize
from utils.network import get_checkpoint_path, IntrinsicCuriosityModule
from models.rand import RandomAgent

FRAME_STACK = 2					# The number of consecutive image states to combine for training a3c on raw images

class RawStack():
	def __init__(self, state_size, num_envs=1, stack_len=FRAME_STACK, load="", gpu=True):
		self.state_size = state_size
		self.stack_len = stack_len
		self.reset(num_envs)

	def reset(self, num_envs, restore=False):
		pass

	def get_state(self, state):
		return state

	def step(self, state, env_action):
		pass

class ImgStack():
	def __init__(self, state_size, num_envs=1, stack_len=FRAME_STACK, load="", gpu=True):
		self.process = lambda x: np.expand_dims(np.transpose(resize(rgb2gray(x) if x.shape[-1] == 3 else x), (2,0,1)), 0)
		self.state_size = [*self.process(np.zeros(state_size)).shape[-2:], (1 if state_size[-1]==3 else state_size[-1])*stack_len]
		self.stack_len = stack_len
		self.reset(num_envs)

	def reset(self, num_envs, restore=False):
		self.num_envs = num_envs
		self.stack = deque(maxlen=self.stack_len)

	def get_state(self, state):
		state = np.concatenate([self.process(s) for s in state]) if self.num_envs > 1 else self.process(state)
		while len(self.stack) < self.stack_len: self.stack.append(state)
		self.stack.append(state)
		return np.concatenate(self.stack, axis=1)

	def step(self, state, env_action):
		pass

class ParallelAgent(RandomAgent):
	def __init__(self, state_size, action_size, agent, num_envs=1, load="", gpu=True, icm=False, **kwargs):
		statemodel = ImgStack if len(state_size) == 3 and type(state_size[0]) == int else RawStack
		self.icm = [IntrinsicCuriosityModule(s_size, a_size) for s_size,a_size in zip(state_size, action_size)] if icm else None
		self.stack = statemodel(state_size, num_envs, load=load, gpu=gpu)
		self.agent = agent(self.stack.state_size, action_size, load=load, gpu=gpu)
		self.state_size = self.stack.state_size
		self.train_rewards = np.zeros([len(self.state_size)])

	def get_env_action(self, env, state, eps=None, sample=True):
		state = self.stack.get_state(state)
		env_action, action = self.agent.get_env_action(env, state, eps, sample)
		return env_action, action, state

	def train(self, state, action, next_state, reward, done):
		self.train_rewards += np.mean(reward, axis=1)
		reward = [icm.train(s,a,ns,r,d) for icm,s,a,ns,r,d in zip(self.icm, state, action, next_state, reward, done)] if self.icm is not None else reward
		self.agent.train(state, action, self.stack.get_state(next_state), reward, done)
		self.eps = self.agent.eps

	def save_model(self, dirname="pytorch", name="best"):
		if hasattr(self.agent, "network"): 
			self.agent.network.save_model(dirname, name)
			if self.icm is not None: [icm.save_model(f"{dirname}/{self.agent.network.name}", f"{name}_{i}") for i,icm in enumerate(self.icm)]

	def load_model(self, dirname="pytorch", name="best"):
		if hasattr(self.agent, "network"): 
			self.agent.network.load_model(dirname, name)
			if self.icm is not None: [icm.load_model(f"{dirname}/{self.agent.network.name}", f"{name}_{i}") for i,icm in enumerate(self.icm)]

	def get_stats(self):
		r_t = self.train_rewards
		self.train_rewards = np.zeros([len(self.state_size)])
		r_i = [icm.get_stats() for icm in self.icm] if self.icm is not None else None
		return {"r_i": r_i, "r_t": r_t, **self.agent.get_stats()}

class DoubleAgent(RandomAgent):
	def __init__(self, state_size, action_size, agent, num_envs=1, load="", gpu=True, agent2=None, **kwargs):
		self.num_envs = num_envs
		self.state_size = state_size
		self.team_split = len(state_size)//2
		self.agent = ParallelAgent(self.state_size[:self.team_split],  action_size[:self.team_split], agent, num_envs, load, gpu, **kwargs)
		self.agent2 = ParallelAgent(self.state_size[self.team_split:],  action_size[self.team_split:], agent2 if agent2 else agent, num_envs, load, gpu)
		self.agents = [self.agent, self.agent2]

	def get_env_action(self, env, state, eps=None, sample=True):
		state = [state[:self.team_split], state[self.team_split:]]
		env_action, action, state = zip(*[agent.get_env_action(env, s, eps, sample) for agent,s in zip(self.agents, state)])
		env_action = [*np.concatenate(env_action)]
		return env_action, action, state

	def train(self, state, action, next_state, reward, done):
		next_state = [next_state[:self.team_split], next_state[self.team_split:]]
		reward = [reward[:self.team_split], reward[self.team_split:]]
		done = [done[:self.team_split], done[self.team_split:]]
		[agent.train(s,a,ns,r,d) for agent,s,a,ns,r,d in zip(self.agents, state, action, next_state, reward, done)]
		self.eps = self.agent.eps

	def save_model(self, dirname="pytorch", name="best"):
		self.agent.save_model(dirname, name)
		if type(self.agent2.agent) != type(self.agent.agent): self.agent2.save_model(dirname, name)

	def get_stats(self):
		return self.agent.get_stats()

class SelfPlayAgent(DoubleAgent):
	def __init__(self, state_size, action_size, agent, num_envs=1, load="", gpu=True, save_freq=1000000, agent2=None, save_dir="pytorch", **kwargs):
		super().__init__(state_size, action_size, agent, num_envs=num_envs, load="", gpu=gpu, agent2=agent, **kwargs)
		self.save_freq = save_freq
		self.save_dir = save_dir
		self.load_model(save_dir)
		self.time = 0

	def train(self, state, action, next_state, reward, done):
		self.agent.train(state[0], action[0], next_state[:self.team_split], reward[:self.team_split], done[:self.team_split])
		self.time += 1
		if self.time % self.save_freq == 0:
			self.save_model(self.save_dir)
			self.load_model(self.save_dir)

	def save_model(self, dirname="pytorch", name=None):
		saves = self.get_save_names(dirname)
		if None not in [dirname, name]: self.agent.save_model(dirname, name)
		else: self.agent.save_model(f"selfplay/{dirname}/save_{len(saves)}", "best")

	def load_model(self, dirname="pytorch", name=None):
		saves = self.get_save_names(dirname)
		if None not in [dirname, name]: self.agent.load_model(dirname, name)
		elif len(saves)>0: 
			choice = np.random.choice(saves, p=np.linspace(0, 2/(len(saves)+1), len(saves)+1)[1:])
			self.agent.load_model(f"selfplay/{dirname}/{saves[-1]}", "best")
			self.agent2.load_model(f"selfplay/{dirname}/{choice}", "best")
			self.v = f"{saves.index(saves[-1])}-{saves.index(choice)}"

	def get_save_names(self, dirname):
		path = os.path.dirname(get_checkpoint_path(self.agent.agent.network.name, f"selfplay/{dirname}"))
		saves = os.listdir(path) if os.path.exists(path) else []
		return sorted(saves, key=lambda x: str(len(x))+x)

	def get_stats(self):
		return {"v": self.v, **super().get_stats()}

class TrainEnv(gym.Wrapper):
	def __init__(self, env):
		super().__init__(env)

	def step(self, action, train=False):
		return super().step(action)

	@property
	def self_play(self):
		return hasattr(self, "num_left_team") and hasattr(self, "num_right_team") and self.num_left_team == self.num_right_team

class TeamEnv(TrainEnv):
	def __init__(self, env):
		super().__init__(env)
		self.action_space, self.observation_space, self.team_space = self.get_spaces()

	def get_spaces(self):
		return self.action_space, self.observation_space, None

	def reset(self, **kwargs):
		observation = self.env.reset(**kwargs)
		return self.split_team(observation)

	def step(self, action, train=False):
		observation, reward, done, info = self.env.step(self.join_team(action))
		return self.split_team(observation), [np.mean(r) for r in self.split_team(reward)], [np.any(done)]*len(set(self.team_space)), info
	
	def split_team(self, item):
		if self.team_space is None: return item
		team_items = [[] for _ in set(self.team_space)]
		for it, team in zip(item, self.team_space):
			team_items[team].append(it)
		return [np.array(x) for x in team_items]

	def join_team(self, item):
		if self.team_space is None: return item
		joined = []
		for it in item:
			it = list(it) if isinstance(it, np.ndarray) else [it]
			joined.extend(it)
		return joined

class ParticleTeamEnv(TeamEnv):
	def get_spaces(self, join_teams=False):
		team_space = []
		state_space = []
		action_space = []
		agents = self.env.agents
		teams = [hasattr(agent, "adversary") and agent.adversary for agent in agents]
		team = -1
		for i,(agent, s_space, a_space) in enumerate(zip(agents, self.observation_space, self.action_space)):
			same_state_space = len(state_space)>0 and s_space.shape == state_space[-1][-1].shape
			same_action_space = len(action_space)>0 and a_space == action_space[-1][-1]
			same_team = i>0 and (hasattr(agent, "adversary") and agent.adversary) == teams[i-1]
			if not all([join_teams, same_state_space, same_action_space, same_team]):
				state_space.append([])
				action_space.append([])
				team += 1
			state_space[-1].append(s_space)
			action_space[-1].append(a_space)
			team_space.append(team)
		for i in range(len(state_space)):
			s = state_space[i]
			state_space[i] = gym.spaces.Box(low=s[0].low[0], high=s[0].high[0], shape=[len(s), *s[0].shape])
		for i in range(len(action_space)):
			a = action_space[i]
			action_space[i] = gym.spaces.MultiDiscrete([a[0].n] * len(a))
		return action_space, state_space, team_space

class FootballTeamEnv(TeamEnv):
	def __init__(self, ggym, env_name, reward_fn=None):
		reps = ["pixels", "pixels_gray", "extracted", "simple115"]
		self.multiagent_args = self.get_multiagent_args(env_name)
		self.reward_fn = reward_fn if reward_fn is not None else None
		rep = reps[3]
		env = ggym.create_environment(env_name=env_name, representation=rep, logdir='/football/logs/', render=rep in reps[:2], **self.multiagent_args)
		super().__init__(env)
		self.eps = 0.2

	@staticmethod
	def get_multiagent_args(env_name):
		match = re.match("^([0-9]+)_vs_([0-9]+)", env_name)
		return {"number_of_left_players_agent_controls": int(match.groups()[0]), "number_of_right_players_agent_controls": int(match.groups()[1])} if match else {}

	def get_spaces(self, split_teams=True):
		team_space = None
		state_space = self.observation_space
		action_space = self.action_space
		self.num_left_team = self.env.unwrapped._agent._num_left_controlled_players
		self.num_right_team = self.env.unwrapped._agent._num_right_controlled_players
		if type(self.env.action_space) == gym.spaces.MultiDiscrete:
			s = self.env.observation_space
			a = self.env.action_space
			team_space = []
			state_space = []
			action_space = []
			action_index = 0
			if self.num_left_team > 0: team_space.extend(range(self.num_left_team) if split_teams else [0]*self.num_left_team)
			if self.num_right_team > 0: team_space.extend(range(self.num_left_team, self.num_left_team+self.num_right_team) if split_teams else [1]*self.num_right_team)
			for team in set(team_space):
				n_players = team_space.count(team)
				state_space.append(gym.spaces.Box(low=np.min(s.low), high=np.max(s.high), shape=[n_players, *s.shape[1:]]))
				action_space.append(gym.spaces.MultiDiscrete(a.nvec[action_index:action_index+n_players]))
				action_index += n_players
		return action_space, state_space, team_space

	def step(self, action, train=False):
		obs, reward, done, info = super().step(action)
		reward = self.reward_fn(obs, reward, self.eps) if self.reward_fn and train else reward
		return obs, reward, done, info

	def reset(self, **kwargs):
		# self.eps *= 0.99
		return super().reset(**kwargs)
