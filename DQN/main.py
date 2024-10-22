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
t.manual_seed(2)
device = t.device('cuda' if t.cuda.is_available() else 'cpu')

reward_per_game = []
max_memory = 700000 
games = 2000  
epsilon = 1 
minibatch_size = 256 
gamma = 0.99
skip_frame = 4 
history_length = 4 
C = 500 
start_learning = 50000 

##########Things still to do###########
# 1. Write class Replay Memory -> Done; Resized images aswell 
# 2. Try a training run ->  Trained for 1000 games with historylength = 8, skipframes = 4 and gamma = 0.99, the network seems to forget what it learns 
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

loss_fn     = nn.SmoothL1Loss()
optimizer   = t.optim.SGD(dqn.parameters(), lr = 0.00025, momentum = 0.95)

def train_replay(memory, minibatch_size):

    states, actions_taken, next_states, action_rewards, terminated = memory.sample()

    main_q_value    = dqn(states.to(device))[actions_taken.to(t.bool).to(device)]
    target_q_value  = t.max(target_dqn(next_states.to(device)).detach(), dim = 1).values
   
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
            action_num = env.action_space.sample()
        else:
            last_state = (memory.create_next_state_hist(-1)).to(device)
            try:
                action_num = t.argmax(dqn(last_state)).item()
            except:
                print('Action-space:', dqn(last_state))
                print('Action-space:', t.argmax(dqn(last_state)))
                print('OG:',last_state.is_cuda)
                print(last_state.to(device).is_cuda)
                print(type(last_state))

        next_state, reward, terminated, truncated, info = env.step(action_num)
        next_state = cv2.resize(next_state, (84, 84), interpolation = cv2.INTER_LINEAR)
        next_state = np.dot(next_state, [0.299, 0.587, 0.114])
        next_state = t.tensor(next_state).view(1, 1, 84, 84).to(t.bfloat16)

        rewards.append(reward)

        reward = t.tensor(reward).to(t.bfloat16)
        termi = t.tensor(not terminated).to(t.bfloat16)
        #There probably is a more simple function for this in pytorch
        action = t.zeros( env.action_space.n)
        action[action_num-1] = 1

        memory.add(state, action, next_state, reward, termi)
        
        state = next_state

        #Train DQN
        if time > start_learning:
            train_replay(memory, minibatch_size)

        #Fixed-update    
        if time % C == 0:
            target_dqn_state_dict   = target_dqn.state_dict() 
            dqn_state_dict          = dqn.state_dict()
            for key in dqn_state_dict:
                target_dqn_state_dict[key] = dqn_state_dict[key]
                target_dqn.load_state_dict(target_dqn_state_dict)    
        time += 1

    print('Memory Size:', memory.size())
    reward_per_game.append(sum(rewards))

    if epsilon > 0.01 and time > start_learning:
        epsilon -= 1/10000
     
torch.save(dqn.state_dict(), 'dqn_weights_2kgames_1.pth')
fig_1, ax_1 = plt.subplots()
ax_1.plot(np.arange(len(reward_per_game)), reward_per_game, 'x')
plt.show()
