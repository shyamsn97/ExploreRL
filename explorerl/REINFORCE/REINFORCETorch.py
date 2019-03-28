import numpy as np
import torch
from explorerl.agents import BaseTorchAgent
from collections import deque
import random

class REINFORCETorch(BaseTorchAgent):
    def __init__(self,estimator=None,gamma=1.0,learning_rate=0.001, featurizer=None,scaler=None,replay_size=500):
        super(REINFORCETorch, self).__init__(estimator,gamma,learning_rate,featurizer,scaler,configs={"softmax"},replay_size=replay_size)
        self.name = "REINFORCETorch"
    
    def initialize_model(self,observation_space,action_space):
        super(REINFORCETorch, self).initialize_model(observation_space,action_space)          

        def loss(predictions,targets):
            return -1*torch.sum(torch.mul(torch.log(predictions),targets))

        self.model["loss"] = loss
        self.model["optimizer"] = torch.optim.Adam(params=self.model["estimator"].parameters(),lr=self.learning_rate,weight_decay=0.0001)

        print("Model Created!")

    def train_policy(self):
        return self.stochastic()
    
    def test_policy(self):
        return self.stochastic()
    
    def stochastic(self):
        def act(obs):
            obs = torch.tensor(obs).float()
            estimator = self.model["estimator"]
            probs = estimator(obs)
            if "continuous" not in self.configs:
                return np.random.choice(self.action_space,p=probs.clone().detach().numpy()[0]) , probs
        return act
                   
    def greedy(self):
        def act(obs):
            obs = torch.tensor(obs).float()
            probs = self.model["estimator"](obs)
            with torch.no_grad():
                if "continuous" not in self.configs:
                    _ , action = torch.tensor(probs).max(1)            
            return int(action), probs
        return act
    
    def episodal_train_iter(self,policy):
        #has experience memory, but only updates 
        obs_arr = []
        reward_arr = []
        loss_fn = self.model["loss"]
        optimizer = self.model["optimizer"]
        for obs, action, next_obs, reward, done in self.experience_replay:
            reward_arr.append(reward)
        dr = self.discount_reward(reward_arr)
        for i in range(len(reward_arr)):
            obs, action, next_obs, reward, done = self.experience_replay[i]
            obs = torch.tensor(obs).float()
            probs = self.model["estimator"](obs)
            with torch.no_grad():
                target = np.zeros((1,self.action_space))
                target[0][action] = dr[i]
                target = torch.tensor(target).float()
            loss = loss_fn(probs,target)
            optimizer.zero_grad()
            loss.backward()  
            optimizer.step()
        self.experience_replay = deque(maxlen=self.replay_size)

        