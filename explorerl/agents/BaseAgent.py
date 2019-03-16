import numpy as np
import matplotlib.pyplot as plt

class BaseAgent():
    def __init__(self,observation_space,action_space,epsilon,decay,gamma, 
                 learning_rate,featurizer,scaler,use_bias):
        self.epsilon = epsilon
        self.decay = decay
        self.action_space = action_space
        self.observation_space = observation_space
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
            return np.concatenate(([1],state))
        return np.expand_dims(state,0) 

    def create_model(self):
        pass
    
    def policy(self):
        pass