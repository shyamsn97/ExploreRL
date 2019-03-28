import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from explorerl.utils.models import *

class BaseTfAgent():
    def __init__(self,estimator,gamma,learning_rate,featurizer,scaler,configs={},replay_size=0):
        tf.keras.backend.clear_session()
        self.name = "BaseTfAgent"
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.featurizer = featurizer
        self.scaler = scaler
        self.model = {}
        self.stats = {"rewards":[],"episodes":[],"num_steps":[]}
        self.replay_size = replay_size
        self.experience_replay = deque(maxlen=replay_size)
        self.epsilon = None
        self.decay = None
        self.estimator = estimator
        self.configs = configs
                
    def save_replay(self,obs,action,next_obs,reward,done):
        self.experience_replay.append([obs,action,next_obs,reward,done])

    def discount_reward(self,reward_arr):
        dr = [0]*(len(reward_arr)+1)
        for i in range(len(reward_arr)-1,-1,-1):
            dr[i] = reward_arr[i] + self.gamma*dr[i+1]
        dr = np.array(dr[:len(reward_arr)])
        return dr
    
    def featurize_state(self, state):
        """
        Returns the featurized representation for a state.
        """
        if self.featurizer:
            if self.scaler:
                state = self.scaler.transform([state])
                featurized = self.featurizer.transform(state)
            else:
                featurized = self.featurizer.transform([state])
            return featurized
        if len(state.shape) == 1:
            return np.expand_dims(state,0) 
        return state

    def reset_stats(self):
        self.stats = {"rewards":[],"episodes":[],"num_steps":[]}

    def record_stats(self,steps,its,rewards):
        self.stats["num_steps"].append(steps)
        self.stats["episodes"].append(its)
        self.stats["rewards"].append(rewards)

    def get_description(self,episode,configs):
        if configs["avg_window"] > 0:
            avg_window = configs["avg_window"]
            avg = np.mean(self.stats["rewards"][::-1][:avg_window])
            num_steps = self.stats["num_steps"][-1]
            return "Epsilon : {}, Num Steps : {}, Avg Reward with Window Size {} : {}".format(self.epsilon,num_steps,avg_window,avg) , avg
        return "Epsilon : {}, Num Steps : {}, reward : {}".format(self.epsilon,num_steps,self.stats["rewards"][-1]), self.stats["rewards"][-1]
    
    def update_hyper_params(self,episode):
        pass

    def initialize_model(self,observation_space,action_space):
        self.observation_space = observation_space
        self.action_space = action_space
        if self.featurizer:
            self.observation_space = [*self.featurizer.transform([np.ones(self.observation_space)]).flatten().shape]
        if self.estimator == None:
            self.model["estimator"] = LinearEstimatorTf(input_space=self.observation_space,output_space=self.action_space,configs=self.configs)
        elif type(self.estimator) == type:
            self.model["estimator"] = self.estimator(input_space=self.observation_space,output_space=self.action_space,configs=self.configs)
        else:
            self.model["estimator"] = self.estimator

    def train_policy(self):
        pass

    def test_policy(self):
        pass

    def episodal_train_iter(self,policy):
        pass

    def train_iter(self,policy,action,values,obs,next_obs,reward,done):
        pass

    def play(env,episodes,steps=10000,display=True):
        policy = test_policy()
        for ep in range(episodes):
            observation = env.reset()
            total_reward = 0
            for t in range(steps):
                observation = agent.featurize_state(observation)
                if display:
                    env.render()
                action, values = policy(observation)
                observation, reward, done, info = env.step(action)
                total_reward += reward
                if done:
                    break
            print("Total reward for episode {}: {}".format(ep,total_reward))
        env.close()


