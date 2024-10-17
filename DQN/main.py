import gymnasium as gym
import math
import time
import torch as t
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import model as model 
from tqdm import tqdm

t.cuda.empty_cache()
t.manual_seed(3)
device = t.device('cuda' if t.cuda.is_available() else 'cpu')

reward_per_game = []
replay_memory = []
length_game = []
memory_max = 10000 
games = 20  
epsilon = 1 
minibatch_size = 128 
gamma = 0.98
skip_frame = 4
C = 10000
start_learning = 50000

##########Things still to do###########
# 1. Write class Replay Memory -> Since we need to be memory efficient we have to save as list and question is whether I should make image smaller
# 2. Try a training run -> Does not work why ???
# 3. Get stuff on GPU properly
# 4. Introduce gradient clipping to stabilize training for other games -> Huber loss ?
# 5. Look into unsqueeze my implementation seems too complicated.

env = gym.make("Pong-v4", render_mode = None, frameskip = skip_frame)

dqn         = model.DQN(env).to(device)
target_dqn  = model.DQN(env).to(device)
#Initialize such that target_dqn == dqn
target_dqn.load_state_dict(dqn.state_dict())

loss_fn     = nn.MSELoss()
optimizer   = t.optim.SGD(dqn.parameters(), lr = 0.00001, momentum = 0.9)

def train_replay(replay_memory, minibatch_size):

    random_choice = np.random.choice(len(replay_memory), minibatch_size)
    states          = t.cat([replay_memory[i][0] for i in random_choice], dim = 0)
    action_rewards  = t.stack([replay_memory[i][1] for i in random_choice], dim = 0)
    next_state      = t.cat([replay_memory[i][2] for i in random_choice], dim = 0)
    terminated      = t.stack([replay_memory[i][3] for i in random_choice], dim = 0)

    main_q_value    = t.max(dqn(states), dim = 1).values
    target_q_value  = t.max(target_dqn(next_state), dim = 1).values
   
    optimizer.zero_grad()
    target = action_rewards + gamma * target_q_value * terminated
    loss = loss_fn(main_q_value, target)
    loss.backward()
    optimizer.step()
    
time = 0
n = 0 
for game in tqdm(range(games)):
    rewards = []
    terminated = False
    state = env.reset()[0]

    state = t.permute(t.unsqueeze(t.tensor(state.astype('float64')), 0), (0, 3, 1, 2)).to(device)
    #Transforming image to greyscale
    state = t.unsqueeze(t.sum(state/3, 1), 0)
    while not terminated:
        p = t.rand(1).item() 
        if p < epsilon: 
            action = env.action_space.sample()
        else:
            action = t.argmax(dqn(state).to(device)).item()

        next_state, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)

        next_state = t.unsqueeze(t.sum(t.permute(t.unsqueeze(t.tensor(next_state.astype('float64')), 0), (0, 3, 1, 2))/3, 1), 0)
    #    print('Next_State:', next_state.shape)

        reward = t.tensor(reward)
        termi = t.tensor(not terminated)

        
        #Here I put states into class Replay Memory

        
        state = next_state

        #Train DQN
        if time > start_learning:
            train_replay(replay_memory, minibatch_size)
        else:
            time += 1

        #Fixed-update    
        if n % C == 0:
            target_dqn_state_dict   = target_dqn.state_dict() 
            dqn_state_dict          = dqn.state_dict()
            for key in dqn_state_dict:
                target_dqn_state_dict[key] = dqn_state_dict[key]
                target_dqn.load_state_dict(target_dqn_state_dict)    
        n += 1


#    print(sum(rewards))
    reward_per_game.append(sum(rewards))
    length_game.append(len(rewards))

    if epsilon > 0.01:
        epsilon -= 1/10000
     
fig_1, ax_1 = plt.subplots()
ax_1.plot(np.arange(len(reward_per_game)), reward_per_game, 'x')
plt.show()
fig_2, ax_2 = plt.subplots()
ax_2.plot(np.arange(len(length_game)), length_game, 'x')
plt.show()
