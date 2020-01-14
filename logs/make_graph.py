import os
import re
import numpy as np
import matplotlib.pylab as plt
from collections import deque

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

def graph_logs(env_name="11_vs_11_stochastic"):
	models = ["ddpg", "ddqn", "ppo", "rand"]
	steps, rewards, averages = zip(*[read_log(f"./{model}/{env_name}/logs_0.txt") for model in models])
	plt.plot(steps[3], rewards[3], color="#CCCCCC", linewidth=0.5)
	plt.plot(steps[0], rewards[0], color="#ADFF2F", linewidth=0.5)
	plt.plot(steps[1], rewards[1], color="#00BFFF", linewidth=0.5)
	plt.plot(steps[2], rewards[2], color="#FF1493", linewidth=0.5)
	plt.plot(steps[3], averages[3], color="#999999", label="Avg Random")
	plt.plot(steps[0], averages[0], color="#008000", label="Avg DDPG")
	plt.plot(steps[1], averages[1], color="#0000CD", label="Avg DDQN")
	plt.plot(steps[2], averages[2], color="#FF0000", label="Avg PPO")
	
	plt.legend(loc="best")
	plt.title(f"Training Rewards")
	plt.xlabel("Step")
	plt.ylabel("Total Score")
	plt.grid(linewidth=0.3, linestyle='-')

def main():
	graph_logs()
	plt.show()

main()