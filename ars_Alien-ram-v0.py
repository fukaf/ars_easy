# require
# pip install gym==0.10.5
# pip install pybullet==2.0.8
# ffmpeg

# Importing the libraries
import os
import numpy as np
import gym
from gym import wrappers
import pybullet_envs

# Setting the Hyper Parameters

class Hp():
    
    def __init__(self):
        self.nb_steps = 2000
        self.episode_length = 10000
        self.learning_rate = 0.02
        self.nb_directions = 10
        self.nb_best_directions = 2
        assert self.nb_best_directions <= self.nb_directions
        self.noise = 0.02
        self.seed = 1
        self.env_name = 'Alien-ram-v0'
        self.v2 = True
        self.M = str(self.nb_steps)+"_"+str(self.episode_length)+"_"+str(self.learning_rate)+"_"+str(self.nb_directions)+"_"+str(self.nb_best_directions)+"_"+str(self.noise)+"_"+str(self.seed)
        if self.v2:
            self.M = self.M + "_v2"
        else:
            self.M = self.M + "_v1"
# Normalizing the states

class Normalizer():
    
    def __init__(self, nb_inputs):
        self.n = np.zeros(nb_inputs)
        self.mean = np.zeros(nb_inputs)
        self.mean_diff = np.zeros(nb_inputs)
        self.var = np.zeros(nb_inputs)
    
    def observe(self, x):
        self.n += 1.
        last_mean = self.mean.copy()
        self.mean += (x - self.mean) / self.n
        self.mean_diff += (x - last_mean) * (x - self.mean)
        self.var = (self.mean_diff / self.n).clip(min = 1e-2)
    
    def normalize(self, inputs):
        obs_mean = self.mean
        obs_std = np.sqrt(self.var)
        if not hp.v2:
            return inputs
        return (inputs - obs_mean) / obs_std

# Building the AI


class Policy():
    
    def __init__(self, input_size, output_size):
        self.theta = np.zeros((output_size, input_size))
    
    def evaluate(self, input, delta = None, direction = None):
        if direction is None:
            return self.theta.dot(input)
        elif direction == "positive":
            return (self.theta + hp.noise*delta).dot(input)
        else:
            return (self.theta - hp.noise*delta).dot(input)
    
    def sample_deltas(self):
        return [np.random.randn(*self.theta.shape) for _ in range(hp.nb_directions)]
    
    def update(self, rollouts, sigma_r):
        step = np.zeros(self.theta.shape)
        for r_pos, r_neg, d in rollouts:
            step += (r_pos - r_neg) * d
        self.theta += hp.learning_rate / (hp.nb_best_directions * sigma_r) * step
        # self.theta += hp.learning_rate / (hp.nb_best_directions) * step
    def save(self, d, step, reward):
        np.savez(d, param = self.theta, step = step, reward = reward, hyperparam = hp.M)

# Exploring the policy on one specific direction and over one episode

def explore(env, normalizer, policy, direction = None, delta = None):
    done = True
    state = env.reset()
    # state = state.flatten()
    done = False
    num_plays = 0.
    sum_rewards = 0
    while not done :
        normalizer.observe(state)
        state = normalizer.normalize(state)
        action = policy.evaluate(state, delta, direction)
        action = np.argmax(action)
        # print(action)
        state, reward, done, _ = env.step(action)
        shift = 0
        # reward = max(min(reward, 1), -1) # ?? why
        sum_rewards += reward - shift
        num_plays += 1
        # state = state.flatten()
    print(num_plays)
    return sum_rewards

# Training the AI

def train(env, policy, normalizer, hp, d):
    total_reward = np.array([])
    for step in range(hp.nb_steps):
        
        # Initializing the perturbations deltas and the positive/negative rewards
        deltas = policy.sample_deltas()
        positive_rewards = [0] * hp.nb_directions
        negative_rewards = [0] * hp.nb_directions
        
        # Getting the positive rewards in the positive directions
        for k in range(hp.nb_directions):
            positive_rewards[k] = explore(env, normalizer, policy, direction = "positive", delta = deltas[k])
        
        # Getting the negative rewards in the negative/opposite directions
        for k in range(hp.nb_directions):
            negative_rewards[k] = explore(env, normalizer, policy, direction = "negative", delta = deltas[k])
        
        # Gathering all the positive/negative rewards to compute the standard deviation of these rewards
        all_rewards = np.array(positive_rewards + negative_rewards)
        sigma_r = all_rewards.std()
        
        # Sorting the rollouts by the max(r_pos, r_neg) and selecting the best directions
        scores = {k:max(r_pos, r_neg) for k,(r_pos,r_neg) in enumerate(zip(positive_rewards, negative_rewards))}
        order = sorted(scores.keys(), key = lambda x:scores[x], reverse = True)[:hp.nb_best_directions]
        rollouts = [(positive_rewards[k], negative_rewards[k], deltas[k]) for k in order]
        
        # Updating our policy
        policy.update(rollouts, sigma_r)
        
        # Printing the final reward of the policy after the update
        reward_evaluation = explore(env, normalizer, policy)
        print('Step:', step, 'Reward:', reward_evaluation)
        total_reward = np.append(total_reward, reward_evaluation)
        policy.save(d, step, total_reward)

# Running the main code

def mkdir(base, name):
    path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path

hp = Hp()
work_dir = mkdir('exp', 'brs')
moni_dir = mkdir(work_dir, 'monitor')
monitor_dir = mkdir(moni_dir, hp.env_name)
trained_dir = mkdir(work_dir, 'trained_policy')
trainedenv_dir = mkdir(trained_dir, hp.env_name)
param_dir = mkdir(monitor_dir, hp.M)
param_trained_dir = mkdir(trainedenv_dir, hp.M)

np.random.seed(hp.seed)
env = gym.make(hp.env_name)
env._max_episode_steps = hp.episode_length
env = wrappers.Monitor(env, param_dir, force = True)
obs = env.observation_space.shape
nb_inputs = obs[0]
nb_outputs = env.action_space.n
print(nb_inputs,nb_outputs)
policy = Policy(nb_inputs, nb_outputs)
normalizer = Normalizer(nb_inputs)
train(env, policy, normalizer, hp, param_trained_dir)