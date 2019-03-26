import numpy as np
import torch
import sys
import sklearn.pipeline
import sklearn.preprocessing
from sklearn.kernel_approximation import RBFSampler
from tqdm import tqdm
from explorerl.agents import BaseAgent
from explorerl.utils.models import FeedForwardNNTorch
from collections import deque

class DQNTorch(BaseAgent):
    def __init__(self,epsilon=1.0, decay= 0.98, gamma=1.0, 
                 learning_rate=0.1, featurizer=None,scaler=None, use_bias = False,has_replay=True,
                 replay_size=2000,replay_batch=32):
        super(DQNTorch, self).__init__(gamma, 
                 learning_rate, featurizer,scaler,use_bias,has_replay)
        self.name = "DQNTorch"
        self.epsilon = epsilon
        self.decay = decay        
        self.experience_replay = deque(maxlen=replay_size)
        self.replay_batch = replay_batch
        self.original_configs = {"epsilon":self.epsilon,"decay":self.decay}
    
    def initialize_replay(self,env):
        c = 0
        while c < self.experience_replay.maxlen:
            obs = env.reset()
            while True:
                action = env.action_space.sample()
                next_obs, reward, done, info = env.step(action)
                self.save_replay(self.featurize_state(obs),action,self.featurize_state(next_obs),reward,done)
                c += 1
                if done:
                    break
            
    def save_replay(self,obs,action,next_obs,reward,done):
        self.experience_replay.append([obs,action,next_obs,reward,done])

    def replay(self,policy):
        minibatch = random.sample(self.experience_replay, self.replay_batch)
        obs_arr = []
        targets = []
        loss_func = self.model["loss"]
        optimizer = self.model["optimizer"]
        for obs, action, next_obs, reward, done in minibatch:
            _, qvals = policy(obs)
            target = reward
            with torch.no_grad():
                next_ac, next_qs = policy(next_obs)
                val = np.max(next_qs.detach().numpy().flatten())
                if done == False:
                    target += self.gamma*(val)
                target_vals = qvals.clone().detach().numpy()
                target_vals[0][action] = target
            loss = loss_func(qvals,torch.tensor(target_vals,requires_grad=False))
            optimizer.zero_grad()  
            loss.backward()
            optimizer.step()
            obs_arr.append(obs)
            targets.append(target_vals)
        obs_arr = np.array(obs_arr)
        targets = np.array(targets)
        return np.array(obs_arr),np.array(tf.stop_gradient(targets))
    
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

        model = FeedForwardNNTorch(input_space,action_space)
        self.model["outputs"] = model
        self.model["loss"] = torch.nn.MSELoss()
        self.model["optimizer"] = torch.optim.Adam(params=self.model["outputs"].parameters(),lr=self.learning_rate,weight_decay=0.0001)
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
            qvals = self.model["outputs"](obs)
            if np.random.random() < self.epsilon:
                return np.random.choice(self.action_space) , qvals
            with torch.no_grad():
                _ , action = torch.tensor(qvals).max(1)            
            return int(action), qvals
        return act
                  
    def greedy(self):
        def act(obs):
            obs = torch.tensor(obs).float()
            qvals = self.model["outputs"](obs)
            with torch.no_grad():
                _ , action = torch.tensor(qvals).max(1)            
            return int(action), qvals
        return act

    def episodal_train_iter(self,policy):
        obs, targets = self.get_replay(training_op,policy)
        _, outs = policy(obs)
        
    def train_iter(self,policy,action,values,obs,next_obs,reward,done):
        self.save_replay(obs,action,next_obs,reward,done)
        obs, targets = self.replay(policy)
        
    def train_iter(self,policy,action,values,obs,next_obs,reward,done):
        qvals = values[0]
        target = reward
        with torch.no_grad():
            next_ac, next_qs = policy(next_obs)
            val = np.max(next_qs.detach().numpy().flatten())
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