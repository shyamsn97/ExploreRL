import sys
import numpy as np
from tqdm import tqdm
from explorerl.utils import *

class Env_Wrapper():
	def __init__(self,env):
		self.env = env

	def train(self,agent,episodes=100,max_len=10000,description_configs={"avg_window":25},early_stop=False,stop_criteria=20,plot=True):
		bar = tqdm(np.arange(episodes))
		observation_space = self.env.observation_space.shape
		action_space = self.env.action_space.n
		agent.initialize_model(observation_space,action_space)
		agent.reset_stats()
		policy = agent.train_policy()
		criteria = 0 #stopping condition
		for episode in bar:
			observation = self.env.reset()
			agent.update_hyper_params(episode)
			rewards = end = losses = 0
			for t in range(max_len):
			    values = policy(observation)
			    action = values[0]
			    next_obs, reward, done, info = self.env.step(action)
			    agent.train_iter(policy,action,values[1:],observation,next_obs,reward,done)
			    rewards += reward
			    end = t
			    if done:
			        break
			    observation = next_obs
			agent.record_stats(end,episode,rewards)
			description, avg = agent.get_description(episode,description_configs)
			bar.set_description(description)
			if early_stop:
				if avg < prev_avg:
					criteria += 1
				if criteria >= stop_criteria:
					break

			prev_avg = avg
		if plot:
			plot_metrics(agent.stats)
		return agent.stats

	def test(self,agent,num_episodes=1,num_steps=20000,display=False,gif=True):
		play_render(self.env,agent,num_episodes,num_steps,display,gif)
