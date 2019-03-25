import numpy as np
import torch
import sys
import sklearn.pipeline
import sklearn.preprocessing
from sklearn.kernel_approximation import RBFSampler
from tqdm import tqdm
from explorerl.agents import BaseAgent
from explorerl.utils.models import LinearEstimatorTorch

class SarsaTorch(BaseAgent):
    def __init__(self,epsilon=1.0, decay= 0.98, gamma=1.0, 
                 learning_rate=0.01, featurizer=None,scaler=None, use_bias = False):
        super(SarsaTorch, self).__init__(gamma, 
                 learning_rate, featurizer,scaler,use_bias)
        self.name = "SarsaTorch"
        self.epsilon = epsilon
        self.decay = decay       
        self.original_configs = {"epsilon":self.epsilon,"decay":self.decay}
        
    def initialize_model(self,observation_space,action_space):
        self.epsilon = self.original_configs["epsilon"]
        self.decay = self.original_configs["decay"]
        self.observation_space = observation_space[0]
        self.action_space = action_space
        input_space = self.observation_space  
        if self.featurizer:
            input_space = self.featurizer.transform([np.ones(self.observation_space)]).flatten().shape[0]
        if self.use_bias:
            input_space += 1

        self.model["output"] = []
        self.model["loss"] = []
        self.model["optimizer"] = []

        for action in range(self.action_space):
            model = LinearEstimatorTorch(input_space,1)
            self.model["output"].append(model) 
            self.model["loss"].append(torch.nn.MSELoss())
            self.model["optimizer"].append(torch.optim.Adam(params=model.parameters(),lr=self.learning_rate,weight_decay=0.0001))
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
            qvals = []
            for i in range(self.action_space):
                qvals.append(self.model["output"][i](obs))
            if np.random.random() < self.epsilon:
                return np.random.choice(self.action_space) , qvals
            with torch.no_grad():
                _, action = torch.tensor(qvals).max(0)
            return int(action), qvals
        return act
                  
    def greedy(self):
        def act(obs):
            obs = torch.tensor(obs).float()
            qvals = []
            for i in range(self.action_space):
                qvals.append(self.model["output"][i](obs))
            with torch.no_grad():
                _ , action = torch.tensor(qvals).max(0)            
            return int(action), qvals
        return act
    
    def train_iter(self,policy,action,values,obs,next_obs,reward,done):
        qvals = values[0]
        with torch.no_grad():
            next_ac, next_qs = policy(next_obs)
            val = float(next_qs[next_ac])
        target = reward
        if done == False:
            target += self.gamma*(val)
        loss_func = self.model["loss"][action]
        optimizer = self.model["optimizer"][action]
        loss = loss_func(qvals[action][0],torch.tensor(target,requires_grad=False))
        optimizer.zero_grad()  
        loss.backward()
        optimizer.step()
        
    def train(self,env,episodes=200,early_stop=False,stop_criteria=20):
        prev_avg = -float('inf')
        orig_epsilon = self.epsilon
        bar = tqdm(np.arange(episodes),file=sys.stdout)
        policy = self.epsilon_greedy()
        criteria = 0 #stopping condition
        for i in bar:
            observation = env.reset()
            self.epsilon *= (self.decay**i)
            rewards = 0
            end = 0
            losses = 0
            for t in range(10000):
                values = policy(observation)
                action = values[0]
                next_obs, reward, done, info = env.step(action)
                self.train_iter(policy,action,values[1:],observation,next_obs,reward,done)
                rewards += reward
                end = t
                if done:
                    break
                observation = next_obs
                
            self.stats["num_steps"].append(end)
            self.stats["episodes"].append(i)
            self.stats["rewards"].append(rewards)
            avg = np.mean(self.stats["rewards"][::-1][:25])
            bar.set_description("Epsilon and reward {} : {}".format(self.epsilon,avg))
            
            if avg < prev_avg:
                criteria += 1
                
            if early_stop:
                if criteria >= stop_criteria:
                    break
                    
            prev_avg = avg
        return self.stats  


