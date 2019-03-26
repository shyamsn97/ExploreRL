import numpy as np
import torch
import sys
import sklearn.pipeline
import sklearn.preprocessing
from sklearn.kernel_approximation import RBFSampler
from tqdm import tqdm
from explorerl.agents import BaseAgent
from explorerl.utils.models import *
from collections import deque
import random

class REINFORCETorch(BaseAgent):
    def __init__(self,gamma=1.0,learning_rate=0.001, featurizer=None,scaler=None,use_bias = False):
        super(REINFORCETorch, self).__init__(gamma,learning_rate,featurizer,scaler,use_bias,has_replay=True)
        self.name = "REINFORCETorch"
    
    def initialize_model(self,observation_space,action_space):
        self.observation_space = observation_space[0]
        self.action_space = action_space
        input_space = self.observation_space  
        if self.featurizer:
            input_space = self.featurizer.transform([np.ones(self.observation_space)]).flatten().shape[0]
        if self.use_bias:
            input_space += 1
        model = LinearEstimatorTorch(input_space,action_space,softmax=True)
        def loss(predictions,targets):
            return -1*torch.sum(torch.mul(torch.log(predictions),targets))
        self.model["outputs"] = model
        self.model["loss"] = loss
        self.model["optimizer"] = torch.optim.Adam(params=self.model["outputs"].parameters(),lr=self.learning_rate,weight_decay=0.0001)
        print("Model Created!")

    def train_policy(self):
        return self.stochastic()
    
    def test_policy(self):
        return self.stochastic()
    
    def stochastic(self):
        def act(obs):
            obs = torch.tensor(obs).float()
            estimator = self.model["outputs"]
            probs = estimator(obs)
            return np.random.choice(self.action_space,p=probs.clone().detach().numpy()[0]) , probs
        return act
                   
    def greedy(self):
        def act(obs):
            obs = torch.tensor(obs).float()
            probs = self.model["outputs"](obs)
            with torch.no_grad():
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
            probs = self.model["outputs"](obs)
            with torch.no_grad():
                target = np.zeros((1,self.action_space))
                target[0][action] = dr[i]
                target = torch.tensor(target).float()
            loss = loss_fn(probs,target)
            optimizer.zero_grad()
            loss.backward()  
            optimizer.step()
        self.experience_replay = deque([])
        