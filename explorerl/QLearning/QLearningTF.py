import numpy as np
import tensorflow as tf
import sys
import sklearn.pipeline
import sklearn.preprocessing
from sklearn.kernel_approximation import RBFSampler
from tqdm import tqdm
from explorerl.agents import BaseAgent
from explorerl.utils.models import LinearEstimatorTf

class QLearningTf(BaseAgent):
    def __init__(self,epsilon=1.0, decay= 0.98, gamma=1.0, 
                 learning_rate=0.01, featurizer=None,scaler=None,use_bias = False):
        super(QLearningTf, self).__init__(gamma, 
                 learning_rate, featurizer,scaler,use_bias)
        tf.keras.backend.clear_session()
        self.name = "QLearningTf"
        self.epsilon = epsilon
        self.decay = decay
        self.original_configs = {"epsilon":self.epsilon,"decay":self.decay}
        
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
       
        model = LinearEstimatorTf(input_space=input_space,output_space=self.action_space)
        self.model["outputs"] = model
        
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
            estimator = self.model["outputs"]
            qvals = estimator(obs)
            if np.random.random() < self.epsilon:
                return np.random.choice(self.action_space) , qvals
            return np.argmax(qvals[0]) , qvals
        return act
                
    def greedy(self):
        def act(obs):
            qvals = []
            estimator = self.model["outputs"]
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
            target[0][action] = reward + self.gamma*np.max(next_qs)
        target = tf.stop_gradient(target)
        training_op(self.model["outputs"],obs,target)
        
    def train(self,env,episodes=200,early_stop=False,stop_criteria=20):
        prev_avg = -float('inf')
        orig_epsilon = self.epsilon
        bar = tqdm(np.arange(episodes),file=sys.stdout)
        policy = self.epsilon_greedy()
        criteria = 0 #stopping condition
        loss = self.model["loss"]
        training_op = self.model["training_op"]
        for i in bar:
            observation = env.reset()
            self.epsilon *= (self.decay**i)
            rewards = 0
            end = 0
            for t in range(10000):
                action , qvals = policy(observation)
                next_obs, reward, done, info = env.step(action)
                rewards += reward
                next_action , next_qs = policy(next_obs)
                target = reward + self.gamma*np.max(next_qs)
                inp = self.featurize_state(observation)
                training_op(self.model["outputs"][action],inp,target)
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