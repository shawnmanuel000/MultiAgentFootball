import os
import re
import numpy as np
import matplotlib.pylab as plt
from collections import deque

gym_envs = ["CartPole-v0", "MountainCar-v0", "Acrobot-v1", "Pendulum-v0", "MountainCarContinuous-v0", "CarRacing-v0", "BipedalWalker-v2", "BipedalWalkerHardcore-v2", "LunarLander-v2", "LunarLanderContinuous-v2"]
gfb_envs = ["academy_empty_goal_close", "academy_empty_goal", "academy_run_to_score", "1_vs_1_easy", "5_vs_5", "11_vs_11_stochastic"]
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
			steps, rewards, averages = read_log(f"./{model}/{env_name}/" + sorted(os.listdir(f"./{model}/{env_name}/"))[-1])
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

def main():
	for env_name in [*gfb_envs]: 
		graph_logs(env_name)

main()