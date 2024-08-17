import gymnasium as gym
import math
import time
import torch as t
import matplotlib.pyplot as plt
from tqdm import tqdm

t.manual_seed(333)

env = gym.make("Taxi-v3", render_mode = "human")
env.reset()

#Parameters
train_episodes = 500
max_steps = 50

#keep track of rewards per run and how much its explores
training_rewards = []
exploration_steps = []

#for now just use a Neural Network to approximate Q-function from the state number
for episode in tqdm(range(train_episodes)):

    break
