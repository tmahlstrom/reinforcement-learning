# transforms an infinite observation space into a finite array using rbf kernel 'exemplars' 
# adaptation from Lazy Programmer Inc. https://deeplearningcourses.com/c/deep-reinforcement-learning-in-python

import numpy as np
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler



class FeatureTransformer:
    def __init__(self, env, n_components=400):
        observation_examples = np.array([env.observation_space.sample() for x in range(5000)])
        scaler = StandardScaler()
        scaler.fit(observation_examples)

        featurizer = FeatureUnion([
                ("rbf1", RBFSampler(gamma=5.0, n_components=n_components)),
                ("rbf2", RBFSampler(gamma=2.0, n_components=n_components)),
                ("rbf3", RBFSampler(gamma=1.0, n_components=n_components)),
                ("rbf4", RBFSampler(gamma=0.5, n_components=n_components))
                ])
        example_features = featurizer.fit_transform(scaler.transform(observation_examples))

        self.dimensions = example_features.shape[1]
        self.scaler = scaler
        self.featurizer = featurizer

    def transform(self, observations):
        scaled = self.scaler.transform(observations)
        return self.featurizer.transform(scaled)