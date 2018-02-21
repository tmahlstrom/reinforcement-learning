# tensorflow deep Q learning with experience replay
# implimentation of Mihn et al. 2013, adaptation from Lazy Programmer Inc. https://deeplearningcourses.com/c/deep-reinforcement-learning-in-python
# Python3
# current implimentation requires finite action space
# works on many atari games, though hyperparamter search likely needed

import copy
import gym
from gym import wrappers
import os
import sys
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.misc import imresize


##########################

# for downsampling the input image
# crop will crop [value] pixels from the identified direction of the input image
# discover appropriate crop values using:
# test_image = env.reset() 
# cropped_image = test_image[TOP_CROP:-BOTTOM_CROP, LEFT_CROP:-RIGHT_CROP] *assuming non zero cropping; if zero, remove bottom and right crop variables
# plt.imshow(cropped_image)
# plt.show() 

IMAGE_RESIZE_X = 80
IMAGE_RESIZE_Y = 80
LEFT_CROP = 5
RIGHT_CROP = 5
TOP_CROP = 15
BOTTOM_CROP = 2
GRAY_SCALE = True

# number of experiences to hold in memory, which are sampled during training
# start size should be ~1/10 memory
# for debugging, make these smaller to quickly get the models running
EXP_MEMORY_SIZE = 400000
EXP_START_SIZE = 40000

# number of in-game steps after which model parameters will be coppied to the target model 
TMODEL_UPDATE_PERIOD = 8000

# number of images used to define the state
STATE_DEPTH = 5


def main():

  conv_layer_sizes = [(32, 8, STATE_DEPTH), (64, 4, 2), (64, 3, 1)]
  hidden_layer_sizes = [512]
  gamma = 0.99
  batch_sz = 32
  num_episodes = 10000
  total_t = 0
  experience_replay_buffer = []
  episode_rewards = np.zeros(num_episodes)

  epsilon = 1.0
  epsilon_min = 0.1
  epsilon_change = (epsilon - epsilon_min) / 500000


  env = gym.envs.make("Alien-v0")
  action_count = env.action_space.n


  model = Deep_Q_Network(
    action_count=action_count,
    conv_layer_sizes=conv_layer_sizes,
    hidden_layer_sizes=hidden_layer_sizes,
    gamma=gamma,
    scope="model")
  target_model = Deep_Q_Network(
    action_count=action_count,
    conv_layer_sizes=conv_layer_sizes,
    hidden_layer_sizes=hidden_layer_sizes,
    gamma=gamma,
    scope="target_model"
  )


  with tf.Session() as sess:
    model.set_session(sess)
    target_model.set_session(sess)
    sess.run(tf.global_variables_initializer())


    print("Creating random experiences before learning...")
    obs = env.reset()
    obs_small = image_downsize(obs)
    state = np.stack([obs_small] * STATE_DEPTH, axis=0)
    assert(state.shape == (STATE_DEPTH, IMAGE_RESIZE_Y, IMAGE_RESIZE_X))
    for i in range(EXP_START_SIZE):

        action = np.random.choice(action_count)
        obs, reward, done, _ = env.step(action)
        next_state = update_state(state, obs)
        assert(state.shape == (STATE_DEPTH, IMAGE_RESIZE_Y, IMAGE_RESIZE_X))
        experience_replay_buffer.append((state, action, reward, next_state, done))

        if done:
            obs = env.reset()
            obs_small = image_downsize(obs)
            state = np.stack([obs_small] * STATE_DEPTH, axis=0)
            assert(state.shape == (STATE_DEPTH, IMAGE_RESIZE_Y, IMAGE_RESIZE_X))
        else:
            state = next_state

    if 'monitor' in sys.argv:
      filename = os.path.basename(__file__).split('.')[0]
      monitor_dir = './' + filename + '_' + str(datetime.now())
      env = wrappers.Monitor(env, monitor_dir)


    print("Beginning to learn...")
    for i in range(num_episodes):

      total_t, episode_reward, duration, num_steps_in_episode, time_per_step, epsilon = play_one(
        env,
        total_t,
        experience_replay_buffer,
        model,
        target_model,
        gamma,
        batch_sz,
        epsilon,
        epsilon_change,
        epsilon_min,
      )
      episode_rewards[i] = episode_reward

      last_100_avg = episode_rewards[max(0, i - 100):i + 1].mean()
      print("Episode:", i,
        "Duration:", duration,
        "Num steps:", num_steps_in_episode,
        "Reward:", episode_reward,
        "Training time per step:", "%.3f" % time_per_step,
        "Avg Reward (Last 100):", "%.3f" % last_100_avg,
        "Epsilon:", "%.3f" % epsilon
      )

      sys.stdout.flush()



#### end main ####


class Deep_Q_Network:
  def __init__(self, action_count, conv_layer_sizes, hidden_layer_sizes, gamma, scope):

    self.action_count = action_count
    self.scope = scope

    with tf.variable_scope(scope):

      self.X = tf.placeholder(tf.float32, shape=(None, STATE_DEPTH, IMAGE_RESIZE_Y, IMAGE_RESIZE_X), name='X')

      self.G = tf.placeholder(tf.float32, shape=(None,), name='G')
      self.actions = tf.placeholder(tf.int32, shape=(None,), name='actions')
      Z = self.X / 255.0
      # note: tensorflow convolution needs the order to be: (num_samples, height, width, 'color')
      Z = tf.transpose(Z, [0, 2, 3, 1])
      for num_output_filters, filtersize, poolsize in conv_layer_sizes:
        Z = tf.contrib.layers.conv2d(
          Z,
          num_output_filters,
          filtersize,
          poolsize,
          activation_fn=tf.nn.relu
        )
      Z = tf.contrib.layers.flatten(Z)
      for M in hidden_layer_sizes:
        Z = tf.contrib.layers.fully_connected(Z, M)

      self.predict_output = tf.contrib.layers.fully_connected(Z, action_count)

      selected_action_values = tf.reduce_sum(
        self.predict_output * tf.one_hot(self.actions, action_count),
        reduction_indices=[1]
      )

      cost = tf.reduce_mean(tf.square(self.G - selected_action_values))

      # other optimizers may do better, requires testing
      self.train_op = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6).minimize(cost)

      self.cost = cost

  def copy_from(self, other):
    mine = [t for t in tf.trainable_variables() if t.name.startswith(self.scope)]
    mine = sorted(mine, key=lambda v: v.name)
    theirs = [t for t in tf.trainable_variables() if t.name.startswith(other.scope)]
    theirs = sorted(theirs, key=lambda v: v.name)

    ops = []
    for p, q in zip(mine, theirs):
      actual = self.session.run(q)
      op = p.assign(actual)
      ops.append(op)

    self.session.run(ops)

  def set_session(self, session):
    self.session = session

  def predict(self, states):
    return self.session.run(self.predict_output, feed_dict={self.X: states})

  def update(self, states, actions, targets):
    c, _ = self.session.run(
      [self.cost, self.train_op],
      feed_dict={
        self.X: states,
        self.G: targets,
        self.actions: actions
      }
    )
    return c

  def sample_action(self, x, eps):
    if np.random.random() < eps:
      return np.random.choice(self.action_count)
    else:
      return np.argmax(self.predict([x])[0])


def play_one(
  env,
  total_t,
  experience_replay_buffer,
  model,
  target_model,
  gamma,
  batch_size,
  epsilon,
  epsilon_change,
  epsilon_min):

  t0 = datetime.now()

  obs = env.reset()
  obs_small = image_downsize(obs)
  state = np.stack([obs_small] * STATE_DEPTH, axis=0)
  assert(state.shape == (STATE_DEPTH, IMAGE_RESIZE_Y, IMAGE_RESIZE_X))
  loss = None


  total_time_training = 0
  num_steps_in_episode = 0
  episode_reward = 0

  done = False
  while not done:

    if total_t % TMODEL_UPDATE_PERIOD == 0:
      target_model.copy_from(model)
      print("Copied model parameters to target model. total_t = %s, period = %s" % (total_t, TMODEL_UPDATE_PERIOD))


    action = model.sample_action(state, epsilon)
    obs, reward, done, _ = env.step(action)
    obs_small = image_downsize(obs)
    next_state = np.append(state[1:], np.expand_dims(obs_small, 0), axis=0)
    assert(state.shape == (STATE_DEPTH, IMAGE_RESIZE_Y, IMAGE_RESIZE_X))



    episode_reward += reward

    if len(experience_replay_buffer) == EXP_MEMORY_SIZE:
      experience_replay_buffer.pop(0)

    experience_replay_buffer.append((state, action, reward, next_state, done))

    t0_2 = datetime.now()
    loss = learn(model, target_model, experience_replay_buffer, gamma, batch_size)
    dt = datetime.now() - t0_2

    total_time_training += dt.total_seconds()
    num_steps_in_episode += 1


    state = next_state
    total_t += 1

    epsilon = max(epsilon - epsilon_change, epsilon_min)

  return total_t, episode_reward, (datetime.now() - t0), num_steps_in_episode, total_time_training/num_steps_in_episode, epsilon


def learn(model, target_model, experience_replay_buffer, gamma, batch_size):

  samples = random.sample(experience_replay_buffer, batch_size)
  states, actions, rewards, next_states, dones = map(np.array, zip(*samples))


  next_Qs = target_model.predict(next_states)
  next_Q = np.amax(next_Qs, axis=1)
  targets = rewards + np.invert(dones).astype(np.float32) * gamma * next_Q


  loss = model.update(states, actions, targets)
  return loss


def image_downsize(original_image):
  (y, x, _) = original_image.shape
  cropped_image = original_image[
    TOP_CROP:(y - BOTTOM_CROP),
    LEFT_CROP:(x - RIGHT_CROP)]
  if GRAY_SCALE:
    cropped_image = cropped_image.mean(axis=2)
  cropped_resized_image = imresize(cropped_image, size=(IMAGE_RESIZE_Y, IMAGE_RESIZE_X), interp='nearest')
  return cropped_resized_image


def update_state(state, obs):
  downsized_obs = image_downsize(obs)
  return np.append(state[1:], np.expand_dims(downsized_obs, 0), axis=0)


####

if __name__ == '__main__':
  main()
