import numpy as np
import torch
import sys
import sklearn.pipeline
import sklearn.preprocessing
from sklearn.kernel_approximation import RBFSampler
from tqdm import tqdm
from explorerl.agents import BaseAgent
from explorerl.utils.models import LinearEstimatorTorch
    
class QLearningTorch(BaseAgent):
    def __init__(self,observation_space,action_space,epsilon=1.0, decay= 0.98, gamma=1.0, 
                 learning_rate=0.01, featurizer=None,scaler=None, use_bias = False):
        super(QLearningTorch, self).__init__(observation_space,action_space,epsilon, decay, gamma, 
                 learning_rate, featurizer,scaler,use_bias)
        self.create_model()
        
        
    def create_model(self):
        input_space = self.observation_space  
        if self.featurizer:
            input_space = self.featurizer.transform([np.arange(self.observation_space)]).flatten().shape[0]
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
    
    def default_policy(self):
        return self.epsilon_greedy()
    
    def epsilon_greedy(self):
        def act(obs):
            obs = torch.tensor(self.featurize_state(obs)).float()
            qvals = []
            for i in range(self.action_space):
                qvals.append(self.model["output"][i](obs))
            if np.random.random() < self.epsilon:
                return np.random.choice(self.action_space) , qvals
            with torch.no_grad():
                _, action = torch.tensor(qvals).max(0)
            return action, qvals
        return act
                  
    def greedy(self):
        def act(obs):
            obs = torch.tensor(self.featurize_state(obs)).float()
            qvals = []
            for i in range(self.action_space):
                qvals.append(self.model["output"][i](obs))
            with torch.no_grad():
                _, action = torch.tensor(qvals).max(0)            
            return int(action), qvals
        return act
    
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
                action , qvals = policy(observation)
                next_obs, reward, done, info = env.step(int(action))
                rewards += reward
                with torch.no_grad():
                    next_ac, next_qs = policy(next_obs)
                    val = float(np.max(next_qs))
                target = reward + self.gamma*(val)
                loss_func = self.model["loss"][action]
                optimizer = self.model["optimizer"][action]
                loss = loss_func(qvals[action],torch.tensor(target,requires_grad=False))
                optimizer.zero_grad()  
                loss.backward()
                optimizer.step()
                losses += loss.item()
                # Adjust weights & reset gradients
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
    
    