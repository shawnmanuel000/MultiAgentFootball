import os
import re
import gym
import cv2
import time
import torch
import random
import inspect
import datetime
import numpy as np
np.set_printoptions(precision=3, sign=' ', floatmode="fixed")

IMG_DIM = 64					# The height and width to scale the environment image to

def rgb2gray(image):
	gray = np.dot(image, [0.299, 0.587, 0.114]).astype(np.float32)
	return np.expand_dims(gray, -1)

def resize(image, scale=0.5):
	dim = (int(image.shape[0]*scale), int(image.shape[1]*scale))
	img = cv2.resize(image, dsize=dim, interpolation=cv2.INTER_CUBIC)
	return np.expand_dims(img, -1) if image.shape[-1]==1 else img

def make_video(imgs, dim, filename):
	video = cv2.VideoWriter(filename, 0, 60, dim)
	for img in imgs:
		video.write(img.astype(np.uint8))
	video.release()

def to_env(env, action):
	action_normal = (1+action)/2
	action_range = env.action_space.high - env.action_space.low
	env_action = env.action_space.low + np.multiply(action_normal, action_range)
	return env_action

def from_env(env, env_action):
	action_range = env.action_space.high - env.action_space.low
	action_normal = np.divide(env_action - env.action_space.low, action_range)
	action = 2*action_normal - 1
	return action

def rollout(env, agent, eps=None, render=False, sample=False):
	state = env.reset()
	total_reward = None
	done = False
	with torch.no_grad():
		while not np.all(done):
			if render: env.render()
			env_action = agent.get_env_action(env, state, eps, sample)[0]
			state, reward, done, _ = env.step(env_action)
			reward = np.array(reward) if type(reward) == list else reward
			total_reward = reward if total_reward is None else total_reward + reward
	return total_reward

class Logger():
	def __init__(self, model_class, label, **kwconfig):
		models = ["ddqn", "ddpg", "ppo", "rand", "coma", "maddpg", "mappo"]
		self.config = kwconfig
		self.label = label
		self.model_class = model_class
		self.model_name = inspect.getmodule(model_class).__name__.split(".")[-1]
		os.makedirs(f"logs/{self.model_name}/{label}/", exist_ok=True)
		self.run_num = len([n for n in os.listdir(f"logs/{self.model_name}/{label}/")])
		self.model_src = [line for line in open(inspect.getabsfile(self.model_class))]
		self.net_src = [line for line in open(f"utils/network.py") if re.match("^[A-Z]", line)] if self.model_name in models else None
		self.trn_src = [line for line in open(f"train.py")] if self.model_name in models else None
		self.log_path = f"logs/{self.model_name}/{label}/logs_{self.run_num}.txt"
		self.log_num = 0

	def log(self, string, stats, debug=True):
		with open(self.log_path, "a+") as f:
			if self.log_num == 0: 
				self.start_time = time.time()
				f.write(f"Model: {self.model_class}, Dir: {self.label}, Date: {datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
				if self.config: f.writelines("\n".join([f"{k}: {v}," for k,v in self.config.items()]) + "\n\n")
				if self.model_src: f.writelines(self.model_src + ["\n"])
				if self.net_src: f.writelines(self.net_src + ["\n"])
				if self.trn_src: f.writelines(self.trn_src + ["\n"])
				f.write("\n")
			f.write(f"{string} <{time.strftime('%d-%H:%M:%S', time.gmtime(time.time()-self.start_time))}> ({{{', '.join([f'{k}: {np.round(v, 3) if v is not None else v}' for k,v in stats.items()])}}})\n")
		if debug: print(string)
		self.log_num += 1

	def get_classes(self, model):
		return [v for k,v in model.__dict__.items() if inspect.getmembers(v)[0][0] == "__class__"]