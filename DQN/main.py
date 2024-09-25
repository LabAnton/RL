import gymnasium as gym
import math
import time
import torch as t
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import model as model 
from tqdm import tqdm

t.manual_seed(333)
device = t.device('cuda' if t.cuda.is_available() else 'cpu')

rewards = []
replay_memory = []
memory_max = 96000
runs = 1000
epsilon = 0.3
minibatch_size = 32


env = gym.make("Pong-v4", render_mode = 'human')
state = env.reset()[0]

dqn = model.DQN(env).to(device)
target_dqn = model.DQN(env)

for run in range(runs):
    p = t.rand(1).item()
    next_state, reward, terminated, truncated, info = env.step(action)
    #Observation is RGB for Pong 3, 210, 160 (C, H, W), using no history yet

    if p < epsilon: 
        action = env.action_space.sample()
    else:
        obs = t.unsqueeze(t.tensor(next_state.astype('float64')), 0)
        action = dqn(t.permute(obs, (0, 3, 1, 2)).to(device))

    rewards.append(reward)
    #No done in pong
    replay_memory.append({'state': s, 'action': a, 'reward': reward, 'next_state': obs})
    
    #change to next state
    state = next_state

def train_replay(replay_memory, minibatch_size):
    minibatch = np.random.choise(replay_memory, minibatch_size)

