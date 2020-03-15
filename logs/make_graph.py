import os
import re
import numpy as np
import matplotlib.pylab as plt
from collections import deque
from mpl_toolkits import mplot3d
np.set_printoptions(precision=3)

gym_envs = ["CartPole-v0", "MountainCar-v0", "Acrobot-v1", "Pendulum-v0", "MountainCarContinuous-v0", "CarRacing-v0", "BipedalWalker-v2", "BipedalWalkerHardcore-v2", "LunarLander-v2", "LunarLanderContinuous-v2"]
gfb_envs = ["academy_empty_goal_close", "academy_empty_goal", "academy_run_to_score", "academy_run_to_score_with_keeper", "academy_single_goal_versus_lazy", "academy_3_vs_1_with_keeper", "1_vs_1_easy", "5_vs_5", "11_vs_11_stochastic"]
env_name = gym_envs[2]

def read_log(path):
	steps = []
	rewards = []
	rolling = []
	averages = deque(maxlen=100)
	with open(path, "r") as f:
		for line in f:
			match = re.match("^Step: (.*), .*Reward: ([^ ]*).*, Avg: ([^ ]*)", line.strip('\n'))
			if match:
				steps.append(float(match.groups()[0]))
				rewards.append(float(match.groups()[1]))
				averages.append(float(match.groups()[1]))
				rolling.append(np.mean(averages, axis=0))
	return steps, rewards, rolling

def read_multiagent_log(path):
	steps = []
	rewards = []
	rolling = []
	averages = deque(maxlen=100)
	for _ in range(100): averages.append(0)
	with open(path, "r") as f:
		for line in f:
			match = re.match("^Step: ([^,]*), Reward: \[ *(-?\d+.\d*)", line.strip('\n'))
			if match:
				steps.append(float(match.groups()[0]))
				rewards.append(float(match.groups()[1]))
				averages.append(float(match.groups()[1]))
				rolling.append(np.mean(averages))
	return steps, rewards, rolling

def graph_logs(env_name=env_name, show=False):
	plt.figure()
	folder = "OpenAI" if env_name in gym_envs else "Football"
	models = ["ppo", "ddpg", "ddqn", "rand"]
	light_cols = {"ddqn":"#ADFF2F", "ddpg":"#00BFFF", "ppo":"#FF1493", "rand":"#CCCCCC"}
	dark_cols = {"ddqn":"#008000", "ddpg":"#0000CD", "ppo":"#FF0000", "rand":"#999999"}
	for model in reversed(models):
		if os.path.exists(f"./{model}/{env_name}/"):
			files = sorted(os.listdir(f"./{model}/{env_name}/"), key=lambda v: len(v) + v)
			steps, rewards, averages = read_log(f"./{model}/{env_name}/" + files[-1])
			plt.plot(steps, rewards, color=light_cols[model], linewidth=0.25, zorder=0)
			plt.plot(steps, averages, color=dark_cols[model], label=f"Avg {model.upper()}", zorder=1)
	try: 
		steps
		plt.legend(loc="best")
		plt.title(f"Training Rewards for {env_name}")
		plt.xlabel("Step")
		plt.ylabel("Total Reward")
		plt.grid(linewidth=0.3, linestyle='-')
		os.makedirs(f"graphs/{folder}/", exist_ok=True)
		print(f"Saving: graphs/{folder}/{env_name}.png")
		plt.savefig(f"graphs/{folder}/{env_name}.png", dpi=1200, pad_inches=0.125)
		if show: plt.show()
	except NameError:
		pass

def graph_multiagent(show=True):
	env_name = "5_vs_5"#"3_vs_3_custom"
	models = ["mappo", "maddpg", "coma", "rand"]
	light_cols = {"coma":"#ADFF2F", "maddpg":"#00BFFF", "mappo":"#FF1493", "rand":"#CCCCCC"}
	dark_cols = {"coma":"#008000", "maddpg":"#0000CD", "mappo":"#FF0000", "rand":"#999999"}
	for model in reversed(models[:1]):
		file_name = f"logs/{model}/{env_name}/"
		if os.path.exists(file_name):
			files = sorted(os.listdir(f"{file_name}/"), key=lambda v: f"{len(v)}-{v}")
			steps, rewards, averages = read_multiagent_log(file_name + files[-1])
			plt.plot(steps, rewards, color=light_cols[model], linewidth=0.25, zorder=0)
			plt.plot(steps, averages, color=dark_cols[model], label=f"Avg {model.upper()}", zorder=1)

	try: 
		steps
		plt.legend(loc="best")
		plt.title(f"Training Rewards for {env_name}")
		plt.xlabel("Step")
		plt.ylabel("Total Reward")
		plt.grid(linewidth=0.3, linestyle='-')
		# os.makedirs(f"graphs/{folder}/", exist_ok=True)
		# print(f"Saving: graphs/{folder}/{env_name}.png")
		# plt.savefig(f"graphs/{folder}/{env_name}.png", dpi=1200, pad_inches=0.125)
		if show: plt.show()
	except NameError:
		pass

def plot_reward_fn():
	x = np.linspace(-1,1,50)
	y = np.linspace(-0.42,0.42,50)
	z = lambda x,y: np.where(x>0, np.maximum(x - np.abs(y)*np.sign(x), 0.5*x), np.minimum(x - np.abs(y)*np.sign(x), 0.5*x))
	X, Y = np.meshgrid(x,y)
	Z = z(X,Y)

	fig = plt.figure()
	ax = fig.gca(projection='3d')
	# ax.plot_surface(X, Y, Z, cmap=plt.cm.RdYlGn)
	ax.contour3D(X, Y, Z, 100, cmap='RdYlGn')
	ax.set_title('Ball Distance from Goal Reward')
	ax.set_xlabel('Length')
	ax.set_ylabel('Width')
	ax.set_zlabel('Ball Reward')
	plt.show()

def plot_exploration(show=False):
	env_name = "5_vs_5"
	light_cols = {0:"#5DFF2F", 2:"#00BFFF", 1:"#FF1493", 3:"#CCCCCC"}
	dark_cols = {0:"#008000", 2:"#0000CD", 1:"#FF0000", 3:"#888888"}
	exp_types = ["Baseline", "Reward Shape", "ICM", "Both"]
	# files = {"maddpg": [2,6,4,7], "mappo": [14,11,13,12], "coma": [12,13,9,14]}
	files = {"maddpg": [2,6,4,7], "mappo": [14,11,13,12], "coma": [12,13,17,14]}
	# files = {"maddpg": [22,23,21,7], "mappo": [24,22,23,12], "coma": [21,19,20,14]}
	# files = {"maddpg": [8,9,10,7], "mappo": [17,18,13,12], "coma": [12,18,9,14]}
	for model in files.keys():
		plt.figure()
		max_avs = []
		path_name = f"logs/{model}/{env_name}/"
		file_names = [f"logs_{x}.txt" for x in files[model]]
		for i,(file_name, name) in enumerate(list(zip(file_names, exp_types))[:-1]):
			path = path_name + file_name
			if os.path.exists(path):
				steps, rewards, averages = read_multiagent_log(path)
				steps = [s/1000 for s in steps]
				plt.plot(steps, rewards, color=light_cols[i], linewidth=0.25, zorder=0)
				plt.plot(steps, averages, color=dark_cols[i], label=f"Avg {name} ({file_name.replace('.txt','')})", zorder=1)
				max_avs.append(list(np.round([averages[-1], np.max(averages)],2)))
				# print(f"{model}, {exp_types[i]}: {np.round(np.max(averages),2)}")
		print("\midrule\n" + model.upper() + " & " + ' & '.join([f"{a}({m})" for a,m in max_avs]) + " \\\\")
		
		try: 
			steps
			plt.legend(loc="best")
			plt.title(f"{model.upper()} Evaluation Rewards for {env_name}")
			plt.xlabel("Step (1000s)")
			plt.ylabel("Total Reward")
			plt.grid(linewidth=0.3, linestyle='-')
			os.makedirs(f"logs/graphs/exploration/", exist_ok=True)
			# print(f"Saving: logs/graphs/exploration/{env_name}_{model}.png")
			plt.savefig(f"logs/graphs/exploration/{env_name}_{model}.png", dpi=600, pad_inches=0.05)
			if show: plt.show()
		except NameError:
			pass


def main():
	# for env_name in [*gfb_envs]: 
	# 	graph_logs(env_name)
	# graph_multiagent()
	# plot_reward_fn()
	plot_exploration()

main()