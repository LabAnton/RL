import gymnasium as gym
import math
import time
import torch as t
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import model as model 
import memory_buffer as mb
import cv2
from tqdm import tqdm

t.cuda.empty_cache()
t.manual_seed(3)
device = t.device('cuda' if t.cuda.is_available() else 'cpu')

reward_per_game = []
replay_memory = []
length_game = []
max_memory = 500000 
games = 1000  
epsilon = 1 
minibatch_size = 128 
gamma = 0.99
skip_frame = 4 
history_length = 8 
C = 10000
start_learning = max_memory/10

##########Things still to do###########
# 1. Write class Replay Memory -> Done; Resized images aswell 
# 2. Try a training run ->  Trained for 1000 games with historylength = 8, skipframes = 4 and gamma = 0.99, the network seems to forget what it learns since it sometimes peaks at -16 total 
#                           reward per game and then goes to -21 again
# 3. Get stuff on GPU properly -> Done
# 4. Introduce gradient clipping to stabilize training for other games -> Huber loss ?
# 5. Look into unsqueeze my implementation seems too complicated.
# 6. Look into ALE documentation. Seems my implementation is old

env = gym.make("Pong-v4", render_mode = None, frameskip = skip_frame)

dqn         = model.DQN(env, history_length).to(t.bfloat16).to(device)
target_dqn  = model.DQN(env, history_length).to(t.bfloat16).to(device)
#Initialize such that target_dqn == dqn
target_dqn.load_state_dict(dqn.state_dict())
memory = mb.Memory_buffer(max_memory, history_length, minibatch_size)

loss_fn     = nn.MSELoss()
optimizer   = t.optim.SGD(dqn.parameters(), lr = 0.00001, momentum = 0.9)

def train_replay(replay_memory, minibatch_size):

    states, next_states, action_rewards, terminated = memory.sample()

    main_q_value    = t.max(dqn(states.to(device)), dim = 1).values
    target_q_value  = t.max(target_dqn(next_states.to(device)), dim = 1).values
   
    optimizer.zero_grad()
    target = action_rewards.to(device) + gamma * target_q_value * terminated.to(device)
    loss = loss_fn(main_q_value, target)
    loss.backward()
    optimizer.step()
    
time = 0
for game in tqdm(range(games)):
    rewards = []
    terminated = False
    state = env.reset()[0]
    #Resize
    state = cv2.resize(state, (84, 84), interpolation = cv2.INTER_LINEAR)
    #Transfrom to greyscale
    state = np.dot(state, [0.299, 0.587, 0.114])
    state = t.tensor(state).view(1, 1, 84, 84).to(t.bfloat16)

    while not terminated:
        p = t.rand(1).item() 
        if p < epsilon: 
            action = env.action_space.sample()
        else:
            #bug here because state has no history, yet.
            last_state = (memory.create_next_state_hist(-1)).to(device)
            try:
                action = t.argmax(dqn(last_state)).item()
            except:
                print('Action-space:', dqn(last_state))
                print('Action-space:', t.argmax(dqn(last_state)))
                print('OG:',last_state.is_cuda)
                print(last_state.to(device).is_cuda)
                print(type(last_state))

        next_state, reward, terminated, truncated, info = env.step(action)
        next_state = cv2.resize(next_state, (84, 84), interpolation = cv2.INTER_LINEAR)
        next_state = np.dot(next_state, [0.299, 0.587, 0.114])
        next_state = t.tensor(next_state).view(1, 1, 84, 84).to(t.bfloat16)

        rewards.append(reward)

        reward = t.tensor(reward).to(t.bfloat16)
        termi = t.tensor(not terminated).to(t.bfloat16)

        memory.add(state, next_state, reward, termi)
        
        state = next_state

        #Train DQN
        if time > start_learning:
            train_replay(replay_memory, minibatch_size)

        #Fixed-update    
        if time % C == 0:
            target_dqn_state_dict   = target_dqn.state_dict() 
            dqn_state_dict          = dqn.state_dict()
            for key in dqn_state_dict:
                target_dqn_state_dict[key] = dqn_state_dict[key]
                target_dqn.load_state_dict(target_dqn_state_dict)    
        time += 1

    reward_per_game.append(sum(rewards))
    length_game.append(len(rewards))

    if epsilon > 0.01 :
        epsilon -= 1/10000
     
fig_1, ax_1 = plt.subplots()
ax_1.plot(np.arange(len(reward_per_game)), reward_per_game, 'x')
plt.show()
fig_2, ax_2 = plt.subplots()
ax_2.plot(np.arange(len(length_game)), length_game, 'x')
plt.show()
