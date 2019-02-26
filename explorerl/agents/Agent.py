import numpy as np
import matplotlib.pyplot as plt

class Agent():
    def __init__(self,env):
        self.env = env
        
    def act(self,observation):
        return self.env.action_space.sample()
    
    def play(self,episode_length=200,episodes=50,silent=True,render=False,plot=False):
        reward_arr = [0]*episodes
        for i_episode in range(episodes):
            observation = self.env.reset()
            for t in range(episode_length):
                if render:
                    self.env.render()
                action = self.act(observation)
                observation, reward, done, info = self.env.step(action)
                reward_arr[i_episode] += reward
                if done:
                    if render:
                        self.env.render()
                    break
            if silent == False:
                print("Episode {} finished with reward: {}".format(i_episode,reward_arr[i_episode]))
                    
        if plot:
            plt.plot(np.arange(episodes)+1,reward_arr,'ro')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.show()