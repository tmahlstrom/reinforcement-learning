# transforms an infinite observation space into a finite array using bins
# adaptation from Lazy Programmer Inc. https://deeplearningcourses.com/c/deep-reinforcement-learning-in-python

import numpy as np


# use to transform an feature vector with continous features into one with discrete featuers
# e.g., transform a continuous location representation into a discrete grid-like representation
class Bin_FeatureTransformer:
    def __init__(self, number_features, bin_per_feature = 9, range_bins = [], bias_bins = []):
        if range_bins == []:
            range_bins = [1 for x in range(number_features)]
        if bias_bins == []:
            bias_bins = [0 for x in range(number_features)]
        assert(len(range_bins) == number_features & len(bias_bins) == number_features)
        
        for bin in range(number_features):
            half_range = range_bins[bin]/2
            bias = bias_bins[bin]
            bin_tag = str(bin)
            self.bin_tag = np.linspace(-half_range + bias, half_range + bias, bin_per_feature)


    def transform(self, observation):
        binned_obs = []

        for n in range(len(observation)):
            bin_tag = str(n)
            binned_obs.append(to_bin(observation[n], self.bin_tag))

        return build_state(binned_obs)


def to_bin(value, bins):
    return np.digitize(x=[value], bins=bins)[0]
    
def build_state(features):
    return int("".join(map(lambda feature: str(int(feature)), features)))