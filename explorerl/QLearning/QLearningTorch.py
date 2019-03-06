import numpy as np
import torch
import sys
import sklearn.pipeline
import sklearn.preprocessing
from sklearn.kernel_approximation import RBFSampler
from tqdm import tqdm

class LinearEstimator(torch.nn.Module):
    def __init__(self,input_space,output_space):
        super(LinearEstimator,self).__init__()
        self.linear = torch.nn.Linear(input_space,output_space)
    
    def forward(self,x):
        x = self.linear(x)
        return x
        
class QLearningTorch():
    def __init__(self, env, epsilon=1.0, decay= 0.98, gamma=1.0, 
                 learning_rate=0.0001, featurize=False, use_bias = False):
        self.epsilon = epsilon
        self.decay = decay
        self.env = env
        self.action_space = env.action_space.n
        self.state_space = env.observation_space.shape[0]
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.featurize = featurize
        self.featurizer = None
        self.scaler = None
        self.model = {}
        self.use_bias = use_bias
        self.create_model()
        self.stats = {"rewards":[],"episodes":[],"num_steps":[]}
    
    def featurize_state(self, state):
        """
        Returns the featurized representation for a state.
        """
        if self.featurize:
            scaled = self.scaler.transform([state])
            featurized = self.featurizer.transform(scaled)
            if self.use_bias:
                return np.concatenate(([1],featurized[0]))
            return featurized[0]
        if self.use_bias:
            return np.concatenate(([1],state))
        return state  
    
    def create_model(self):
        input_space = self.state_space    
        # featurizing code taken from https://github.com/dennybritz/reinforcement-learning/tree/master/FA
        # Used to convert a state to a featurizes representation.
        # Use RBF kernels with different variances to cover different parts of the space
        if self.featurize:
            input_space = 400
            observation_examples = np.array([self.env.observation_space.sample() for x in range(10000)])
            self.scaler = sklearn.preprocessing.StandardScaler()
            self.scaler.fit(observation_examples)

            self.featurizer = sklearn.pipeline.FeatureUnion([
                    ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
                    ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
                    ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
                    ("rbf4", RBFSampler(gamma=0.5, n_components=100))
                    ])
            self.featurizer.fit(observation_examples)
        if self.use_bias:
            input_space += 1
        self.model["output"] = []
        self.model["loss"] = []
        self.model["optimizer"] = []
        for action in range(self.action_space):
            model = LinearEstimator(input_space,1)
            self.model["output"].append(model) 
            self.model["loss"].append(torch.nn.MSELoss())
            #deriv = state*(1/2*(target-output))
            self.model["optimizer"].append(torch.optim.Adam(params=model.parameters(),lr=self.learning_rate,weight_decay=0.0001))
        print("Model Created!")
    
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
    
    def train(self,episodes=200,early_stop=False,stop_criteria=20):
        prev_avg = -float('inf')
        orig_epsilon = self.epsilon
        bar = tqdm(np.arange(episodes),file=sys.stdout)
        policy = self.epsilon_greedy()
        criteria = 0 #stopping condition
        for i in bar:
            observation = self.env.reset()
            self.epsilon *= (self.decay**i)
            rewards = 0
            end = 0
            losses = 0
            for t in range(10000):
                action , qvals = policy(observation)
                next_obs, reward, done, info = self.env.step(int(action))
                rewards += reward
                with torch.no_grad():
                    next_acc, next_qs = policy(next_obs)
                    val = float(np.max(next_qs))
                target = reward + self.gamma*(val)
                loss_func = self.model["loss"][action]
                optimizer = self.model["optimizer"][action]
                loss = loss_func(torch.tensor(target),qvals[action])
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
    
    