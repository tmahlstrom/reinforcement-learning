# tensorflow policy gradient method using RBF kernels 
# adaptation from Lazy Programmer Inc. https://deeplearningcourses.com/c/deep-reinforcement-learning-in-python
# Python3
# RBF kernels allow it to work with an infinite action space
# currently implimented for MountainCar game

import gym
from gym import wrappers
import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
from rbf_feature_transformer import FeatureTransformer
from RL_plots import plot_cost_to_go, plot_running_avg


# discount on the value of future states
GAMMA = 0.95 
EPISODES = 3


# creates the environment and the features used to navigate the infinite action space
# creates and initializes the policy and value models
# runs updates to the models 
def main():
  env = gym.make('MountainCarContinuous-v0')
  rbf_features = FeatureTransformer(env, n_components=100)
  dims = rbf_features.dimensions
  policy_model = PolicyModel(dims, rbf_features, [])
  value_model = ValueModel(dims, rbf_features, [])
  init = tf.global_variables_initializer()
  session = tf.InteractiveSession()
  session.run(init)
  policy_model.set_session(session)
  value_model.set_session(session)

  if 'monitor' in sys.argv:
    filename = os.path.basename(__file__).split('.')[0]
    monitor_dir = './' + filename + '_' + str(datetime.now())
    env = wrappers.Monitor(env, monitor_dir)

  totalrewards = np.empty(EPISODES)
  costs = np.empty(EPISODES)
  for n in range(EPISODES):
    totalreward, num_steps = play_one_td(env, policy_model, value_model)
    totalrewards[n] = totalreward
    if n % 1 == 0:
      print("episode:", n, "total reward: %.1f" % totalreward, "num steps: %d" % num_steps, "avg reward (last 100): %.1f" % totalrewards[max(0, n-100):(n+1)].mean())

  print("avg reward for last 100 episodes:", totalrewards[-100:].mean())

  plt.plot(totalrewards)
  plt.title("Rewards")
  plt.show()

  plot_running_avg(totalrewards)
  plot_cost_to_go(env, value_model)


class HiddenLayer:
  def __init__(self, M1, M2, activation=tf.nn.tanh, use_bias=True, zeros=False):
    if zeros:
      W = np.zeros((M1, M2), dtype=np.float32)
    else:
      W = tf.random_normal(shape=(M1, M2)) * np.sqrt(2. / M1, dtype=np.float32)
    self.W = tf.Variable(W)

    self.use_bias = use_bias
    if use_bias:
      self.b = tf.Variable(np.zeros(M2).astype(np.float32))

    self.activation = activation

  def forward(self, X):
    if self.use_bias:
      a = tf.matmul(X, self.W) + self.b
    else:
      a = tf.matmul(X, self.W)
    return self.activation(a)


# used to estimate pi(a|s)
class PolicyModel:
  def __init__(self, D, features, hidden_layer_sizes=[]):
    self.features = features

    # create graph
    M1 = D
    self.layers = []
    for M2 in hidden_layer_sizes:
      layer = HiddenLayer(M1, M2)
      self.layers.append(layer)
      M1 = M2

    # final layer mean and varaince, format of output given infinite action space
    self.mean_layer = HiddenLayer(M1, 1, lambda x: x, use_bias=False, zeros=True)
    self.stdv_layer = HiddenLayer(M1, 1, tf.nn.softplus, use_bias=False, zeros=False)

    # inputs and targets
    self.X = tf.placeholder(tf.float32, shape=(None, D), name='X')
    self.actions = tf.placeholder(tf.float32, shape=(None,), name='actions')
    self.advantages = tf.placeholder(tf.float32, shape=(None,), name='advantages')

    # get final layer
    Z = self.X
    for layer in self.layers:
      Z = layer.forward(Z)

    # calculate output and cost
    mean = self.mean_layer.forward(Z)
    stdv = self.stdv_layer.forward(Z) + 1e-5 # smoothing

    mean = tf.reshape(mean, [-1])
    stdv = tf.reshape(stdv, [-1]) 

    norm = tf.contrib.distributions.Normal(mean, stdv)
    self.predict_op = tf.clip_by_value(norm.sample(), -1, 1)

    log_probs = norm.log_prob(self.actions)
    cost = -tf.reduce_sum(self.advantages * log_probs + 0.1*norm.entropy())
    self.train_op = tf.train.AdamOptimizer(1e-3).minimize(cost)

  def set_session(self, session):
    self.session = session

  def partial_fit(self, X, actions, advantages):
    X = np.atleast_2d(X)
    X = self.features.transform(X)
    
    actions = np.atleast_1d(actions)
    advantages = np.atleast_1d(advantages)
    self.session.run(
      self.train_op,
      feed_dict={
        self.X: X,
        self.actions: actions,
        self.advantages: advantages,
      }
    )

  def predict(self, X):
    X = np.atleast_2d(X)
    X = self.features.transform(X)
    return self.session.run(self.predict_op, feed_dict={self.X: X})

  def sample_action(self, X):
    p = self.predict(X)[0]
    return p


# approximates V(s), model very similar to policy model
class ValueModel:
  def __init__(self, D, features, hidden_layer_sizes=[]):
    self.features = features
    self.costs = []

    self.layers = []
    M1 = D
    for M2 in hidden_layer_sizes:
      layer = HiddenLayer(M1, M2)
      self.layers.append(layer)
      M1 = M2

    layer = HiddenLayer(M1, 1, lambda x: x)
    self.layers.append(layer)

    self.X = tf.placeholder(tf.float32, shape=(None, D), name='X')
    self.Y = tf.placeholder(tf.float32, shape=(None,), name='Y')

    Z = self.X
    for layer in self.layers:
      Z = layer.forward(Z)
    Y_hat = tf.reshape(Z, [-1])
    self.predict_op = Y_hat

    cost = tf.reduce_sum(tf.square(self.Y - Y_hat))
    self.cost = cost
    self.train_op = tf.train.AdamOptimizer(1e-1).minimize(cost)

  def set_session(self, session):
    self.session = session

  def partial_fit(self, X, Y):
    X = np.atleast_2d(X)
    X = self.features.transform(X)
    Y = np.atleast_1d(Y)
    self.session.run(self.train_op, feed_dict={self.X: X, self.Y: Y})
    cost = self.session.run(self.cost, feed_dict={self.X: X, self.Y: Y})
    self.costs.append(cost)

  def predict(self, X):
    X = np.atleast_2d(X)
    X = self.features.transform(X)
    return self.session.run(self.predict_op, feed_dict={self.X: X})


def play_one_td(env, pmodel, vmodel):
  observation = env.reset()
  done = False
  totalreward = 0
  iters = 0

  while not done and iters < 2000:
    action = pmodel.sample_action(observation)
    prev_observation = observation
    observation, reward, done, info = env.step([action])

    totalreward += reward

    # update the models
    V_next = vmodel.predict(observation)
    G = reward + GAMMA*V_next
    advantage = G - vmodel.predict(prev_observation)
    pmodel.partial_fit(prev_observation, action, advantage)
    vmodel.partial_fit(prev_observation, G)

    iters += 1

  return totalreward, iters



####


if __name__ == '__main__':
  main()

