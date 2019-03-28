import numpy as np
import torch
import sys
import sklearn.pipeline
import sklearn.preprocessing
from sklearn.kernel_approximation import RBFSampler
from tqdm import tqdm
from explorerl.agents import BaseTorchAgent
from explorerl.utils.models import *

class SarsaTorch(BaseTorchAgent):
    def __init__(self,estimator=None,epsilon=1.0, decay= 0.98, gamma=1.0, 
                 learning_rate=0.01, featurizer=None,scaler=None):
        super(SarsaTorch, self).__init__(estimator,gamma,learning_rate, featurizer,scaler,configs={},replay_size=0)
        self.name = "SarsaTorch"
        self.epsilon = epsilon
        self.decay = decay        
        
    def initialize_model(self,observation_space,action_space):
        super(SarsaTorch, self).initialize_model(observation_space,action_space)          

        self.model["loss"] = torch.nn.MSELoss()
        self.model["optimizer"] = torch.optim.Adam(params=self.model["estimator"].parameters(),lr=self.learning_rate,weight_decay=0.0001)
        print("Model Created!")
    
    def update_hyper_params(self,episode):
        self.epsilon *= (self.decay**episode)
        
    def train_policy(self):
        return self.epsilon_greedy()
    
    def test_policy(self):
        return self.greedy()
    
    def epsilon_greedy(self):
        def act(obs):
            obs = torch.tensor(obs).float()
            qvals = self.model["estimator"](obs)
            if np.random.random() < self.epsilon:
                return np.random.choice(self.action_space) , qvals
            with torch.no_grad():
                _ , action = torch.tensor(qvals).max(1)            
            return int(action), qvals
        return act
                  
    def greedy(self):
        def act(obs):
            obs = torch.tensor(obs).float()
            qvals = self.model["estimator"](obs)
            with torch.no_grad():
                _ , action = torch.tensor(qvals).max(1)            
            return int(action), qvals
        return act
    
    def train_iter(self,policy,action,values,obs,next_obs,reward,done):
        qvals = values[0]
        target = reward
        with torch.no_grad():
            next_ac, next_qs = policy(next_obs)
            val = next_qs.detach().numpy().flatten()[next_ac]
            if done == False:
                target += self.gamma*(val)
            target_vals = qvals.clone().detach().numpy()
            target_vals[0][action] = target
        loss_func = self.model["loss"]
        optimizer = self.model["optimizer"]
        loss = loss_func(qvals,torch.tensor(target_vals,requires_grad=False))
        optimizer.zero_grad()  
        loss.backward()
        optimizer.step()


