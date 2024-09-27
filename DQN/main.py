import gymnasium as gym
import math
import time
import torch as t
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import model as model 
from tqdm import tqdm

t.manual_seed(33)
device = t.device('cpu')

rewards = []
replay_memory = []
memory_max = 96000
runs = 50 
epsilon = 0.8 
minibatch_size = 32
gamma = 0.9


env = gym.make("Pong-v4", render_mode = 'human')
state = env.reset()[0]

#Network is overparameterized because I get RGB, I have to check later what to do
dqn = model.DQN(env).to(device)
target_dqn = model.DQN(env).to(device)
loss_fn = nn.MSELoss()
optimizer = t.optim.SGD(dqn.parameters(), lr = 0.0001, momentum = 0.9)

#Very inefficient and a lot of lines, can write it more beautifully
def train_replay(replay_memory, minibatch_size):

    states = t.stack([t.tensor(replay_memory[i][0]) for i in np.random.choice(len(replay_memory), minibatch_size)], dim = 0).to(device)
    action_rewards = t.stack([t.tensor(replay_memory[i][2]) for i in np.random.choice(len(replay_memory), minibatch_size)], dim = 0)
    next_state = t.stack([t.tensor(replay_memory[i][3]) for i in np.random.choice(len(replay_memory), minibatch_size)], dim = 0)
    print(action_rewards.shape)
    pred = dqn(t.permute(states, (0, 3, 1, 2)).to(device))
    pred_2 = t.tensor(t.max(dqn(t.permute(states, (0, 3, 1, 2))), dim = 1).values)
    target_pred = target_dqn(t.permute(states, (0, 3, 1, 2)).to(device))

#    print('Pred:', pred)
#    print(pred_2)
#    print('Target_pred:', target_pred.shape)
#    print('Max Target', t.max(target_pred, dim = 1).values)
#    print('Action Rewards:', action_rewards)
    
    optimizer.zero_grad()
    target = action_rewards + gamma * t.max(target_pred, dim = 1).values
    loss = loss_fn(pred_2, target)
    print(loss)
    loss.backward()
    optimizer.step()
#    print('Final Target:', target)

for run in range(runs):
    p = t.rand(1).item()
    #Observation is RGB for Pong 3, 210, 160 (C, H, W), using no history yet

    if p < epsilon: 
        action = env.action_space.sample()
    else:
        obs = t.unsqueeze(t.tensor(state.astype('float64')), 0)
        action = t.argmax(dqn(t.permute(obs, (0, 3, 1, 2)).to(device))).item()

    next_state, reward, terminated, truncated, info = env.step(action)
    rewards.append(reward)
    #Replay_memory right now is infinite
    replay_memory.append([state.astype('float64'), action, reward, next_state.astype('float64')])
    
    #change to next state
    state = next_state

    train_replay(replay_memory, minibatch_size)
    #Have no Fixed Q-Training implemented
    
    if epsilon > 0.01:
        epsilon -= 1/100
    

    

