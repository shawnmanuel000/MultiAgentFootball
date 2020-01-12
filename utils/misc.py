import os
import re
import gym
import cv2
import torch
import random
import inspect
import numpy as np
import matplotlib.pyplot as plt

IMG_DIM = 64					# The height and width to scale the environment image to

def rgb2gray(image):
	gray = np.dot(image, [0.299, 0.587, 0.114]).astype(np.float32)
	return gray

def resize(image, dim=IMG_DIM):
	img = cv2.resize(image, dsize=(dim,dim), interpolation=cv2.INTER_CUBIC)
	return img

def show_image(img, filename="test.png", save=True):
	if save: plt.imsave(filename, img)
	plt.imshow(img, cmap=plt.get_cmap('gray'))
	plt.show()

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
	total_reward = 0
	done = False
	with torch.no_grad():
		while not done:
			if render: env.render()
			env_action = agent.get_env_action(env, state[np.newaxis,:], eps, sample)[0]
			env_action = env_action[0] if hasattr(env.action_space, "n") else env_action.reshape(-1)
			state, reward, done, _ = env.step(env_action)
			total_reward += reward
	return total_reward

class Logger():
	def __init__(self, model_class, load_dir, **kwconfig):
		self.config = kwconfig
		self.load_dir = load_dir
		self.model_class = model_class
		self.model_name = inspect.getmodule(model_class).__name__.split(".")[-1]
		os.makedirs(f"logs/{self.model_name}/{load_dir}/", exist_ok=True)
		self.run_num = len([n for n in os.listdir(f"logs/{self.model_name}/{load_dir}/")])
		self.model_src = [line for line in open(inspect.getabsfile(self.model_class))]
		self.net_src = [line for line in open(f"utils/network.py") if re.match("^[A-Z]", line)] if self.model_name in ["ddpg", "ppo"] else None
		self.trn_src = [line for line in open(f"train_a3c.py")] if self.model_name in ["ddpg", "ppo"] else None
		self.log_path = f"logs/{self.model_name}/{load_dir}/logs_{self.run_num}.txt"
		self.log_num = 0

	def log(self, string, debug=True):
		with open(self.log_path, "a+") as f:
			if self.log_num == 0: 
				f.write(f"Model: {self.model_class}, Dir: {self.load_dir}\n")
				if self.config: f.writelines(" ".join([f"{k}: {v}," for k,v in self.config.items()]) + "\n\n")
				if self.model_src: f.writelines(self.model_src + ["\n"])
				if self.net_src: f.writelines(self.net_src + ["\n"])
				if self.trn_src: f.writelines(self.trn_src + ["\n"])
				f.write("\n")
			f.write(f"{string}\n")
		if debug: print(string)
		self.log_num += 1

	def get_classes(self, model):
		return [v for k,v in model.__dict__.items() if inspect.getmembers(v)[0][0] == "__class__"]