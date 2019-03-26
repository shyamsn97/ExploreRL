import numpy as np
import tensorflow as tf
import sys
import sklearn.pipeline
import sklearn.preprocessing
from sklearn.kernel_approximation import RBFSampler
from tqdm import tqdm
from explorerl.agents import BaseAgent
from explorerl.utils.models import *
from collections import deque
import random

class REINFORCETf(BaseAgent):
    def __init__(self,gamma=1.0,learning_rate=0.001, featurizer=None,scaler=None,use_bias = False):
        super(REINFORCETf, self).__init__(gamma,learning_rate,featurizer,scaler,use_bias,has_replay=True)
        tf.keras.backend.clear_session()
        self.name = "REINFORCETf"
    
    def initialize_model(self,observation_space,action_space):
        self.observation_space = observation_space[0]
        self.action_space = action_space
        input_space = self.observation_space  
        if self.featurizer:
            input_space = self.featurizer.transform([np.ones(self.observation_space)]).flatten().shape[0]
        if self.use_bias:
            input_space += 1
       
        model = LinearEstimatorTf(input_space=input_space,output_space=self.action_space,softmax=True)
        self.model["outputs"] = model
        
        def log_loss(model,predictions,targets):
            return -1*(tf.reduce_sum(tf.multiply(tf.math.log(predictions),targets))) + tf.add_n(model.losses)
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        def train_step(model,inputs,targets):
            with tf.GradientTape() as tape:
                predictions = model(inputs)
                total_loss = log_loss(model,predictions,targets)
            gradients = tape.gradient(total_loss,model.trainable_variables)
            optimizer.apply_gradients(zip(gradients,model.trainable_variables))
                
        self.model["loss"] = log_loss
        self.model["training_op"] = train_step
        print("Model Created!")

    def train_policy(self):
        return self.stochastic()
    
    def test_policy(self):
        return self.stochastic()
    
    def stochastic(self):
        def act(obs):
            estimator = self.model["outputs"]
            probs = estimator(obs)
            return np.random.choice(self.action_space,p=np.array(probs[0])) , probs
        return act
    
    def greedy(self):
        def act(obs):
            estimator = self.model["outputs"]
            probs = estimator(obs)
            return np.argmax(probs[0]) , probs
        return act
    
    def episodal_train_iter(self,policy):
        #has experience memory, but only updates 
        obs_arr = []
        reward_arr = []
        training_op = self.model["training_op"]
        for obs, action, next_obs, reward, done in self.experience_replay:
            reward_arr.append(reward)
        dr = self.discount_reward(reward_arr)
        for i in range(len(reward_arr)):
            obs, action, next_obs, reward, done = self.experience_replay[i]
            target = np.zeros((1,self.action_space))
            target[0][action] = dr[i]
            training_op(self.model["outputs"],obs,target)
        self.experience_replay = deque([])
        