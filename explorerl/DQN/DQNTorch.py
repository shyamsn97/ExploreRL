import numpy as np
import torch
from explorerl.agents import BaseTorchAgent
from collections import deque
import random

class DQNTorch(BaseTorchAgent):
    def __init__(self,estimator=None,epsilon=1.0, decay=0.995, gamma=.95, 
                 learning_rate=0.0001, featurizer=None,scaler=None,replay_size=2000,replay_batch=64):
        super(DQNTorch, self).__init__(estimator,gamma,learning_rate, featurizer,scaler,configs={},replay_size=replay_size)
        self.name = "DQNTorch"
        self.epsilon = epsilon
        self.decay = decay        
        self.replay_batch = replay_batch
    
    def initialize_model(self,observation_space,action_space):
        super(DQNTorch, self).initialize_model(observation_space,action_space)          

        self.model["loss"] = torch.nn.MSELoss()
        self.model["optimizer"] = torch.optim.Adam(params=self.model["estimator"].parameters(),lr=self.learning_rate,weight_decay=0.0001)
        print("Model Created!")

    def replay(self,policy):
        minibatch = random.sample(self.experience_replay, self.replay_batch)
        loss_func = self.model["loss"]
        optimizer = self.model["optimizer"]
        for obs, action, next_obs, reward, done in minibatch:
            _, qvals = policy(obs)
            target = reward
            with torch.no_grad():
                _, next_qs = policy(next_obs)
                val = np.max(next_qs.detach().numpy().flatten())
                if not done:
                    target = reward + self.gamma*(val)
                target_vals = qvals.clone().detach().numpy()
                target_vals[0][action] = target
                target_vals = torch.tensor(target_vals,requires_grad=False)
            optimizer.zero_grad()  
            loss = loss_func(qvals,target_vals)
            loss.backward()
            optimizer.step()

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

    def episodal_train_iter(self,policy):
        self.replay(policy)
        
    def train_iter(self,policy,action,values,obs,next_obs,reward,done):
        self.replay(policy)

        
        
