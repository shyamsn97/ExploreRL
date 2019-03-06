import numpy as np
import tensorflow as tf
import sys
import sklearn.pipeline
import sklearn.preprocessing
from sklearn.kernel_approximation import RBFSampler
from tqdm import tqdm

class QLearningTF():
    def __init__(self, env, epsilon=1.0, decay= 0.98, gamma=1.0, 
                 learning_rate=0.01, featurize=False, use_bias = False):
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
        tf.reset_default_graph()
        self.create_model()
        self.sess = tf.Session()
        self.stats = {"rewards":[],"episodes":[],"num_steps":[]}
    
    def featurize_state(self, state):
        """
        Returns the featurized representation for a state.
        """
        if self.featurize:
            if self.scaler:
                state = self.scaler.transform([state])
            featurized = self.featurizer.transform(state)
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
            self.featurizer.fit(self.scaler.transform(observation_examples))
        if self.use_bias:
            input_space += 1
        self.model["X"] = tf.placeholder(shape=[None,input_space],dtype=tf.float64)
        self.model["target"] = tf.placeholder(shape=(None),dtype=tf.float64)
        self.model["optimizer"] = tf.train.AdamOptimizer(self.learning_rate)
        self.model["outputs"] = []
        self.model["losses"] = []
        self.model["training_ops"] = []
        self.model["Ws"] = []
        self.model["bs"] = []
        
        for action in range(self.action_space):
            
            W = tf.Variable(tf.truncated_normal([input_space, 1], mean=0.0, 
                                            stddev=1.0, dtype=tf.float64), name="W_"+str(action))
            output = tf.matmul(self.model["X"],W)
            
            #l2 regularization
            loss = tf.reduce_mean(tf.squared_difference(self.model["target"],output)) + \
                                    0.0001*(tf.reduce_sum(tf.square(W)))
            optimizer = tf.train.AdamOptimizer(self.learning_rate) #deriv = state*(1/2*(target-output))
            training_op = optimizer.minimize(loss,var_list=[W])
            
            self.model["Ws"].append(W)
            self.model["outputs"].append(output)
            self.model["losses"].append(loss)
            self.model["training_ops"].append(training_op)        
        print("Model Created!")
    
    def epsilon_greedy(self):
        def act(obs):
            qvals = []
            for action in range(self.action_space):
                estimator = self.model["outputs"][action]
                qval = self.sess.run(estimator,feed_dict={self.model["X"]:[self.featurize_state(obs)]})[0]
                qvals.append(qval)
            if np.random.random() < self.epsilon:
                return np.random.choice(self.action_space) , qvals
            return np.argmax(qvals) , qvals
        return act
                
    
    def greedy(self):
        def act(obs):
            qvals = []
            for action in range(self.action_space):
                estimator = self.model["outputs"][action]
                qval = self.sess.run(estimator,feed_dict={self.model["X"]:[self.featurize_state(obs)]})[0]
                qvals.append(qval)
            return np.argmax(qvals) , qvals
        return act
    
    def train(self,episodes=200,early_stop=False,stop_criteria=20):
        import sys
        init = tf.global_variables_initializer()
        self.sess.run(init)
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
            for t in range(10000):
                action , qvals = policy(observation)
                next_obs, reward, done, info = self.env.step(action)
                rewards += reward
                next_action , next_qs = policy(next_obs)
                target = reward + self.gamma*np.max(next_qs)
                update = self.model["training_ops"][action]
                self.sess.run(update,feed_dict={self.model["X"]:[self.featurize_state(observation)],
                                                self.model["target"]:[target]})
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
    
    def assign(self,weights):
        self.sess.run(self.model["W"].assign(weights[0]))
