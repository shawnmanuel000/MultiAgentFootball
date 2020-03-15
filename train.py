import gym
import argparse
import numpy as np
import particle_envs.make_env as pgym
import football.gfootball.env as ggym
from models.ppo import PPOAgent
from models.sac import SACAgent
from models.ddqn import DDQNAgent
from models.ddpg import DDPGAgent
from models.rand import RandomAgent
from multiagent.coma import COMAAgent
from multiagent.maddpg import MADDPGAgent
from multiagent.mappo import MAPPOAgent
from utils.wrappers import ParallelAgent, DoubleAgent, SelfPlayAgent, ParticleTeamEnv, FootballTeamEnv, TrainEnv
from utils.envs import EnsembleEnv, EnvManager, EnvWorker, MPI_SIZE, MPI_RANK
from utils.misc import Logger, rollout
np.set_printoptions(precision=3)

gym_envs = ["CartPole-v0", "MountainCar-v0", "Acrobot-v1", "Pendulum-v0", "MountainCarContinuous-v0", "CarRacing-v0", "BipedalWalker-v2", "BipedalWalkerHardcore-v2", "LunarLander-v2", "LunarLanderContinuous-v2"]
gfb_envs = ["academy_empty_goal_close", "academy_empty_goal", "academy_run_to_score", "academy_run_to_score_with_keeper", "academy_single_goal_versus_lazy", "academy_3_vs_1_with_keeper", "1_vs_1_easy", "3_vs_3_custom", "5_vs_5", "11_vs_11_stochastic", "test_example_multiagent"]
ptc_envs = ["simple_adversary", "simple_speaker_listener", "simple_tag", "simple_spread", "simple_push"]
env_name = gym_envs[0]
env_name = gfb_envs[-3]
# env_name = ptc_envs[-2]

def make_env(env_name=env_name, log=False, render=False, reward_shape=False):
	if env_name in gym_envs: return TrainEnv(gym.make(env_name))
	if env_name in ptc_envs: return ParticleTeamEnv(pgym.make_env(env_name))
	ballr = lambda x,y: (np.maximum if x>0 else np.minimum)(x - np.abs(y)*np.sign(x), 0.5*x)
	reward_fn = lambda obs,reward,eps: [0.1*(ballr(o[0,88], o[0,89]) - o[0,96]) + r for o,r in zip(obs,reward)]
	return FootballTeamEnv(ggym, env_name, reward_fn if reward_shape else None)

def train(model, steps=10000, ports=16, env_name=env_name, trial_at=10000, save_at=10, checkpoint=True, save_best=False, log=True, render=False, reward_shape=False, icm=False):
	envs = (EnvManager if type(ports) == list or MPI_SIZE > 1 else EnsembleEnv)(lambda: make_env(env_name, reward_shape=reward_shape), ports)
	agent = (DoubleAgent if envs.env.self_play else ParallelAgent)(envs.state_size, envs.action_size, model, envs.num_envs, load="", gpu=True, agent2=RandomAgent, save_dir=env_name, icm=icm) 
	logger = Logger(model, env_name, num_envs=envs.num_envs, state_size=agent.state_size, action_size=envs.action_size, action_space=envs.env.action_space, envs=type(envs), reward_shape=reward_shape, icm=icm)
	states = envs.reset(train=True)
	total_rewards = []
	for s in range(steps+1):
		env_actions, actions, states = agent.get_env_action(envs.env, states)
		next_states, rewards, dones, _ = envs.step(env_actions, train=True)
		agent.train(states, actions, next_states, rewards, dones)
		states = next_states
		if s%trial_at == 0:
			rollouts = rollout(envs, agent, render=render)
			total_rewards.append(np.mean(rollouts, axis=-1))
			save_dir = env_name + "/" +  "_".join(["rs"]*int(reward_shape) + ["icm"]*int(icm))
			if checkpoint and len(total_rewards) % save_at==0: agent.save_model(save_dir, "checkpoint")
			if save_best and np.all(total_rewards[-1] >= np.max(total_rewards, axis=-1)): agent.save_model(env_name)
			if log: logger.log(f"Step: {s:7d}, Reward: {total_rewards[-1]} [{np.std(rollouts):4.3f}], Avg: {np.mean(total_rewards, axis=0)} ({agent.eps:.4f})", agent.get_stats())

def trial(model, env_name, render, reward_shape=False, icm=False):
	envs = EnsembleEnv(lambda: make_env(env_name, log=True, render=render), 1)
	load_dir = env_name + "/" +  "_".join(["rs"]*int(reward_shape) + ["icm"]*int(icm))
	agent = (DoubleAgent if envs.env.self_play else ParallelAgent)(envs.state_size, envs.action_size, model, gpu=False, load=load_dir, agent2=RandomAgent, save_dir=env_name)
	print(f"Reward: {rollout(envs, agent, eps=0.0, render=True)}")
	envs.close()

def parse_args():
	parser = argparse.ArgumentParser(description="A3C Trainer")
	parser.add_argument("--workerports", type=int, default=[16], nargs="+", help="The list of worker ports to connect to")
	parser.add_argument("--selfport", type=int, default=None, help="Which port to listen on (as a worker server)")
	parser.add_argument("--model", type=str, default="coma", help="Which reinforcement learning algorithm to use")
	parser.add_argument("--steps", type=int, default=200000, help="Number of steps to train the agent")
	parser.add_argument("--reward_shape", action="store_true", help="Whether to shape rewards for football")
	parser.add_argument("--icm", action="store_true", help="Whether to use intrinsic motivation")
	parser.add_argument("--render", action="store_true", help="Whether to render during training")
	parser.add_argument("--trial", action="store_true", help="Whether to show a trial run")
	parser.add_argument("--env", type=str, default="", help="Name of env to use")
	return parser.parse_args()

if __name__ == "__main__":
	args = parse_args()
	env_name = env_name if args.env not in [*gym_envs, *gfb_envs, *ptc_envs] else args.env
	models = {"ddpg":DDPGAgent, "ppo":PPOAgent, "sac":SACAgent, "ddqn":DDQNAgent, "maddpg":MADDPGAgent, "mappo":MAPPOAgent, "coma":COMAAgent, "rand":RandomAgent}
	model = models[args.model] if args.model in models else RandomAgent
	if args.selfport is not None or MPI_RANK>0:
		EnvWorker(self_port=args.selfport, make_env=make_env).start()
	elif args.trial:
		trial(model=model, env_name=env_name, render=args.render, reward_shape=args.reward_shape, icm=args.icm)
	else:
		train(model=model, steps=args.steps, ports=args.workerports[0] if len(args.workerports)==1 else args.workerports, env_name=env_name, render=args.render, reward_shape=args.reward_shape, icm=args.icm)
