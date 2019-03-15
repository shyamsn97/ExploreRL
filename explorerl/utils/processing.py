import numpy as np
import sklearn.pipeline
import sklearn.preprocessing
from sklearn.kernel_approximation import RBFSampler


class BlankScaler():
    def transform(self,obs):
        return obs
    
def create_scaler_featurizer(env,make_scaler=True):
    # featurizing code taken from https://github.com/dennybritz/reinforcement-learning/tree/master/FA
    # Used to convert a state to a featurizes representation.
    # Use RBF kernels with different variances to cover different parts of the space
    observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
    scaler = None
    featurizer = sklearn.pipeline.FeatureUnion([
            ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
            ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
            ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
            ("rbf4", RBFSampler(gamma=0.5, n_components=100))
            ])
    if make_scaler:
        scaler = sklearn.preprocessing.StandardScaler()
        scaler.fit(observation_examples)
        observation_examples = scaler.transform(observation_examples)
    
    featurizer.fit(observation_examples)

    return scaler,featurizer
    