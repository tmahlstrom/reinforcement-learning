# python implimentation of temporal difference method with a lambda eligibility trace
# adaptation from Lazy Programmer Inc. https://deeplearningcourses.com/c/deep-reinforcement-learning-in-python
# Python3
# currently implimented for MountainCar game

import numpy as np
import gym
from gym import wrappers
import os
import sys
from datetime import datetime
from rbf_feature_transformer import FeatureTransformer
import matplotlib.pyplot as plt
from RL_plots import plot_cost_to_go, plot_running_avg


GAMMA = 0.99
LAMBDA = 0.7
EPISODES = 300



def main():
    env = gym.make('MountainCar-v0')
    rbf_features = FeatureTransformer(env)
    model = Model(env, rbf_features)

    if 'monitor' in sys.argv:
        filename = os.path.basename(__file__).split('.')[0]
        monitor_dir = './' + filename + '_' + str(datetime.now())
        env = wrappers.Monitor(env, monitor_dir)

    totalrewards = np.empty(EPISODES)
    costs = np.empty(EPISODES)
    for n in range(EPISODES):
        # eps = 1.0/(0.1*n+1)
        eps = 0.1*(0.97**n)
        # eps = 0.5/np.sqrt(n+1)
        totalreward = play_one(model, env, eps)
        totalrewards[n] = totalreward
        print("episode:", n, "total reward:", totalreward)
    print("avg reward for last 100 episodes:", totalrewards[-100:].mean())
    print("total steps:", -totalrewards.sum())

    plt.plot(totalrewards)
    plt.title("Rewards")
    plt.show()

    plot_running_avg(totalrewards)

    # plot the optimal state-value function
    plot_cost_to_go(env, model)




class Model:
  def __init__(self, env, rbf_feature_transformer):
    self.env = env
    self.models = []
    self.rbf_feature_transformer = rbf_feature_transformer

    dims = rbf_feature_transformer.dimensions
    self.eligibilities = np.zeros((env.action_space.n, dims))
    for i in range(env.action_space.n):
      model = BaseModel(dims)
      self.models.append(model)

  def predict(self, s):
    X = self.rbf_feature_transformer.transform([s])
    assert(len(X.shape) == 2) 
    result = np.stack([m.predict(X) for m in self.models]).T
    assert(len(result.shape) == 2)
    return result

  def update(self, s, a, G):
    X = self.rbf_feature_transformer.transform([s])
    assert(len(X.shape) == 2)
    self.eligibilities *= GAMMA*LAMBDA
    self.eligibilities[a] += X[0]
    self.models[a].partial_fit(X[0], G, self.eligibilities[a])

  def sample_action(self, s, eps):
    if np.random.random() < eps:
      return self.env.action_space.sample()
    else:
      return np.argmax(self.predict(s))



class BaseModel:
  def __init__(self, D):
    self.w = np.random.randn(D) / np.sqrt(D)

  def partial_fit(self, input_, target, eligibility_trace, lr=1e-2):
    self.w += lr*(target - input_.dot(self.w))*eligibility_trace

  def predict(self, X):
    X = np.array(X)
    return X.dot(self.w)



def play_one(model, env, eps):
  observation = env.reset()
  done = False
  totalreward = 0
  iters = 0
  # while not done and iters < 200:
  while not done and iters < 10000:
    action = model.sample_action(observation, eps)
    prev_observation = observation
    observation, reward, done, _ = env.step(action)

    # update the model
    next = model.predict(observation)
    assert(next.shape == (1, env.action_space.n))
    G = reward + GAMMA*np.max(next[0])
    model.update(prev_observation, action, G)

    totalreward += reward
    iters += 1

  return totalreward




####


if __name__ == '__main__':
  main()



