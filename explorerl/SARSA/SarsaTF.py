import numpy as np
import tensorflow as tf
import sys
import sklearn.pipeline
import sklearn.preprocessing
from sklearn.kernel_approximation import RBFSampler
from tqdm import tqdm
from explorerl.agents import BaseTfAgent
from explorerl.utils.models import *

class SarsaTf(BaseTfAgent):
    def __init__(self,estimator=None,epsilon=1.0, decay= 0.98, gamma=1.0,learning_rate=0.01, featurizer=None,scaler=None):
        super(SarsaTf, self).__init__(estimator,gamma,learning_rate, featurizer,scaler,configs={},replay_size=0)
        self.name = "SarsaTf"
        self.epsilon = epsilon
        self.decay = decay
        
    def initialize_model(self,observation_space,action_space):
        super(SarsaTf, self).initialize_model(observation_space,action_space)          

        
        def mse_loss(model,predictions,targets):
            return tf.losses.mean_squared_error(targets,predictions) + tf.add_n(model.losses)
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        def train_step(model,inputs,targets):
            with tf.GradientTape() as tape:
                predictions = model(inputs)
                total_loss = mse_loss(model,predictions,targets)
            gradients = tape.gradient(total_loss,model.trainable_variables)
            optimizer.apply_gradients(zip(gradients,model.trainable_variables))
                
        self.model["loss"] = mse_loss
        self.model["training_op"] = train_step
        print("Model Created!")
    
    def update_hyper_params(self,episode):
        self.epsilon *= (self.decay**episode)
        
    def train_policy(self):
        return self.epsilon_greedy()
    
    def test_policy(self):
        return self.greedy()
    
    def epsilon_greedy(self):
        def act(obs):
            estimator = self.model["estimator"]
            qvals = estimator(obs)
            if np.random.random() < self.epsilon:
                return np.random.choice(self.action_space) , qvals
            return np.argmax(qvals[0]) , qvals
        return act
                
    def greedy(self):
        def act(obs):
            qvals = []
            estimator = self.model["estimator"]
            qvals = estimator(obs)
            return np.argmax(qvals[0]) , qvals
        return act
    
    def train_iter(self,policy,action,values,obs,next_obs,reward,done):
        training_op = self.model["training_op"]
        qvals = values[0]
        next_action , next_qs = policy(next_obs)
        target = np.array(qvals)
        target[0][action] = reward
        if done == False:
            target[0][action] = reward + self.gamma*np.array(next_qs)[0][next_action]
        target = tf.stop_gradient(target)
        training_op(self.model["estimator"],obs,target)
        
