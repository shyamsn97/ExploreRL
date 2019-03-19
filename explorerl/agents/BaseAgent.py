import numpy as np
import matplotlib.pyplot as plt

class BaseAgent():
    def __init__(self,gamma, 
                 learning_rate,featurizer,scaler,use_bias):
        self.name = "Base"
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.featurizer = featurizer
        self.scaler = scaler
        self.model = {}
        self.use_bias = use_bias
        self.stats = {"rewards":[],"episodes":[],"num_steps":[]}
 
    def featurize_state(self, state):
        """
        Returns the featurized representation for a state.
        """
        if self.featurizer:
            if self.scaler:
                state = self.scaler.transform([state])
            featurized = self.featurizer.transform(state)
            if self.use_bias:
                return np.expand_dims(np.concatenate(([1],featurized[0])),0)
            return featurized
        if self.use_bias:
            return np.expand_dims(np.concatenate(([1],state)),0)
        return np.expand_dims(state,0) 

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
            return "Epsilon : {}, avg reward with window size {} : {}".format(self.epsilon,avg_window,avg) , avg
        return "Epsilon and reward {} : {}".format(self.epsilon,self.stats["rewards"][-1]), self.stats["rewards"][-1]
    
    def update_hyper_params(self,episode):
        pass
        
    def initialize_model(self):
        pass
    
    def train_policy(self):
        pass

    def test_policy(self):
        pass

    def train_iter(self,policy,action,values,obs,next_obs,reward,done):
        pass


