import numpy as np
import tensorflow as tf
import sys
import sklearn.pipeline
import sklearn.preprocessing
from sklearn.kernel_approximation import RBFSampler
from tqdm import tqdm
from explorerl.agents import BaseAgent
from explorerl.utils.models import FeedForwardNNTF
from collections import deque
import random

class DQNTF(BaseAgent):
    def __init__(self,epsilon=1.0, decay= 0.995, gamma=.95, 
                 learning_rate=0.0001, featurizer=None,scaler=None,use_bias = False,has_replay=True,
                 replay_size=2000,replay_batch=32):
        super(DQNTF, self).__init__(gamma, 
                 learning_rate, featurizer,scaler,use_bias,has_replay)
        tf.keras.backend.clear_session()
        self.name = "DQNTF"
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
        
        model = FeedForwardNNTF(input_space,self.action_space)
        def mse_loss(model,predictions,targets):
            return tf.reduce_mean(tf.square(tf.subtract(predictions,targets)))
#             return tf.reduce_mean(tf.square(tf.subtract(predictions,targets))) + tf.add_n(model.losses)
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.model["outputs"] = model
        
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
            qvals = self.model["outputs"](obs)
            if np.random.random() < self.epsilon:
                return np.random.choice(self.action_space) , qvals
            return np.argmax(qvals[0]) , qvals
        return act
                
    def greedy(self):
        def act(obs):
            qvals = self.model["outputs"](obs)
            return np.argmax(qvals[0]) , qvals
        return act
    
    def replay(self,training_op,policy):
        minibatch = random.sample(self.experience_replay, self.replay_batch)
        obs_arr = []
        targets = []
        for obs, action, next_obs, reward, done in minibatch:
            reward = reward if not done else -10
            target = reward
            obs_arr.append(obs)
            if not done:
                next_qvals = self.model["outputs"](next_obs)
                target = (reward + self.gamma * np.max(np.array(next_qvals)))
            qvals =  self.model["outputs"](obs)
            target_f = np.array(qvals)
            target_f[0][action] = target
            target_f = tf.stop_gradient(target_f)
            training_op(self.model["outputs"],obs,target_f)
            targets.append(target_f)
        obs_arr = np.array(obs_arr)
        targets = np.array(targets)
        return np.array(obs_arr),np.array(tf.stop_gradient(targets))
    
    def episodal_train_iter(self,policy):
        training_op = self.model["training_op"]
        model = self.model["outputs"]
        obs, targets = self.get_replay(training_op,policy)
        _, outs = policy(obs)
#         training_op(self.model["outputs"],obs,targets)
        
    def train_iter(self,policy,action,values,obs,next_obs,reward,done):
        training_op = self.model["training_op"]
        self.save_replay(obs,action,next_obs,reward,done)
        obs, targets = self.replay(training_op,policy)
#         training_op(self.model["outputs"],obs,targets)
    
    def train(self,env,episodes=200,early_stop=False,stop_criteria=20):
        prev_avg = -float('inf')
        orig_epsilon = self.epsilon
        bar = tqdm(np.arange(episodes),file=sys.stdout)
        policy = self.epsilon_greedy()
        criteria = 0 #stopping condition
        training_op = self.model["training_op"]
        for i in bar:
            observation = env.reset()
            self.epsilon *= (self.decay**i)
            rewards = 0
            end = 0
            observation = self.featurize_state(observation)
            for t in range(10000):
                action , qvals = policy(observation)
                next_obs, reward, done, info = env.step(action)
                rewards += reward
                next_obs = self.featurize_state(next_obs)
                next_action , next_qs = policy(next_obs)
                target = reward + self.gamma*np.max(next_qs[0])
                inp = self.featurize_state(observation)
                training_op(self.model["outputs"],inp,tf.stop_gradient(target))
                end = t
                if done:
                    break
                observation = next_obs
                
            self.stats["num_steps"].append(end)
            self.stats["episodes"].append(i)
            self.stats["rewards"].append(rewards)
            avg = np.mean(self.stats["rewards"][::-1][:25])
            bar.set_description("Epsilon and Num Steps {} : {}".format(self.epsilon,end))
            
            if avg < prev_avg:
                criteria += 1
                
            if early_stop:
                if criteria >= stop_criteria:
                    break
                    
            prev_avg = avg
        return self.stats