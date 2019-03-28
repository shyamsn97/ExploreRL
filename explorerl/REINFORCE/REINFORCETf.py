import numpy as np
import tensorflow as tf
from explorerl.agents import BaseTfAgent
from collections import deque
import random

class REINFORCETf(BaseTfAgent):
    def __init__(self,estimator=None,gamma=1.0,learning_rate=0.001, featurizer=None,scaler=None,replay_size=500):
        super(REINFORCETf, self).__init__(estimator,gamma,learning_rate,featurizer,scaler,configs={"softmax"},replay_size=replay_size)
        self.name = "REINFORCETf"
    
    def initialize_model(self,observation_space,action_space):
        super(REINFORCETf, self).initialize_model(observation_space,action_space)          

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
            estimator = self.model["estimator"]
            probs = estimator(obs)
            if "continuous" not in self.configs:
                return np.random.choice(self.action_space,p=np.array(probs[0])) , probs
        return act
    
    def greedy(self):
        def act(obs):
            estimator = self.model["estimator"]
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
            training_op(self.model["estimator"],obs,target)
        self.experience_replay = deque(maxlen=self.replay_size)


        