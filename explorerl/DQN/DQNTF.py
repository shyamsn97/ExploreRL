import numpy as np
import tensorflow as tf
from explorerl.agents import BaseTfAgent
from collections import deque
import random

class DQNTf(BaseTfAgent):
    def __init__(self,estimator=None,epsilon=1.0, decay= 0.995, gamma=.95,learning_rate=0.0001, featurizer=None,scaler=None,replay_size=2000,replay_batch=32):
        super(DQNTf, self).__init__(estimator,gamma,learning_rate, featurizer,scaler,configs={},replay_size=replay_size)
        self.name = "DQNTf"
        self.epsilon = epsilon
        self.decay = decay
        self.replay_batch = replay_batch

    def initialize_model(self,observation_space,action_space):
        super(DQNTf, self).initialize_model(observation_space,action_space)          

        def mse_loss(model,predictions,targets):
            return tf.reduce_mean(tf.square(tf.subtract(predictions,targets))) + tf.add_n(model.losses)
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        
        def train_step(model,inputs,targets):
            with tf.GradientTape() as tape:
                predictions = model(inputs)
                total_loss = mse_loss(model,predictions,tf.stop_gradient(targets))
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
            qvals = self.model["estimator"](obs)
            if np.random.random() < self.epsilon:
                return np.random.choice(self.action_space) , qvals
            return np.argmax(qvals[0]) , qvals
        return act
                
    def greedy(self):
        def act(obs):
            qvals = self.model["estimator"](obs)
            return np.argmax(qvals[0]) , qvals
        return act
    
    def replay(self,policy):
        minibatch = random.sample(self.experience_replay, self.replay_batch)
        training_op = self.model["training_op"]
        for obs, action, next_obs, reward, done in minibatch:
            target = reward
            if not done:
                next_qvals = self.model["estimator"](next_obs)
                target = (reward + self.gamma * np.max(np.array(next_qvals)))
            qvals =  self.model["estimator"](obs)
            target_f = np.array(qvals)
            target_f[0][action] = target
            training_op(self.model["estimator"],obs,target_f)
    
    def episodal_train_iter(self,policy):
        training_op = self.model["training_op"]
        model = self.model["estimator"]
        obs, targets = self.get_replay(training_op,policy)
        _, outs = policy(obs)
        
    def train_iter(self,policy,action,values,obs,next_obs,reward,done):
        self.replay(policy)

