import gfootball.env as gym

env = gym.create_environment(env_name="academy_empty_goal_close", representation="simple115", stacked=False, logdir='/football/logs/', render=False)
print(f"State space: {env.observation_space.shape}")
print(f"Action space: {env.action_space.n}")

for ep in range (10):
	state = env.reset()
	total_rewards = 0
	done = False
	while not done:
		action = env.action_space.sample()
		state, reward, done, info = env.step(action)
		total_rewards += reward
	print(f"Ep: {ep}, Reward: {total_rewards}")