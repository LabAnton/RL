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
reward_per_game = []
replay_memory = []
losses = []
memory_max = 96000
games = 2 
epsilon = 0.1 
minibatch_size = 32
gamma = 0.9

##########Things still to do###########
# 1. Change RGB to Greyscale or increase conv such that last layer is smaller
# 2. Rewrite train_replay, such that it is better to read and more efficient -> need to build mask
# 3. Set-up Fixed-Q target -> Done
# 4. Try a training run 
# 5. Hyperparameter study
# 6. Get everything on GPU
# 7. Have not thought about terminal state yet.
# 8. Introduce gradient clipping to stabilize training
# 9. Look at Huber loss for other games

env = gym.make("Pong-v4", render_mode = 'human')

#Network is overparameterized because I get RGB, I have to check later what to do
dqn         = model.DQN(env).to(device)
target_dqn  = model.DQN(env).to(device)
#Initialize such that target_dqn == dqn
target_dqn.load_state_dict(dqn.state_dict())
loss_fn     = nn.MSELoss()
optimizer   = t.optim.SGD(dqn.parameters(), lr = 0.0001, momentum = 0.9)

# Still need to build mask
def train_replay(replay_memory, minibatch_size):
    random_choice = np.random.choice(len(replay_memory), minibatch_size)
    print(np.random.choice(len(replay_memory), minibatch_size))
    states          = t.cat([replay_memory[i][0] for i in random_choice], dim = 0)
    action_rewards  = t.stack([replay_memory[i][2] for i in random_choice], dim = 0)
    next_state      = t.cat([replay_memory[i][3] for i in random_choice], dim = 0)

#    print('States:', states.shape)
#    print('Next_States:', next_state.shape)
#    print('Action_rewards:', action_rewards.shape)

    main_q_value    = t.max(dqn(states), dim = 1). values
    target_q_value  = t.max(target_dqn(next_state), dim = 1).values
#    print('Main_q_value:', main_q_value.shape, 'Target_q_value:', target_q_value.shape)
   
    optimizer.zero_grad()
    target = action_rewards + gamma * target_q_value
    loss = loss_fn(main_q_value , target)
    losses.append(loss.item())
    loss.backward()
    optimizer.step()
    

for game in tqdm(range(games)):
    terminated   = False
    state = env.reset()[0]
    
    state = t.permute(t.unsqueeze(t.tensor(state.astype('float64')), 0), (0, 3, 1, 2))
    while not terminated:
        p = t.rand(1).item()
        #Observation is RGB for Pong 3, 210, 160 (C, H, W), using no history yet

        if p < epsilon: 
            action = env.action_space.sample()
        else:
            action = t.argmax(dqn(state)).item()

        next_state, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        next_state = t.permute(t.unsqueeze(t.tensor(next_state.astype('float64')), 0), (0, 3, 1, 2))
        reward = t.tensor(reward)

        if len(replay_memory) > memory_max:
            replay_memory.pop(0)

        replay_memory.append([state, action, reward, next_state])
        
        state = next_state

        #Train DQN
        train_replay(replay_memory, minibatch_size)

    reward_per_game.append(sum(rewards)/len(rewards))

    #Fixed-update    
    target_dqn_state_dict   = target_dqn.state_dict() 
    dqn_state_dict          = dqn.state_dict()
    for key in dqn_state_dict:
        target_dqn_state_dict[key] = dqn_state_dict[key]
    target_dqn.load_state_dict(target_dqn_state_dict)    

    if epsilon > 0.01:
        epsilon -= 1/1000
     
fig_1, ax_1 = plt.subplots()
ax_1.plot(np.arange(len(reward_per_game)), reward_per_game, 'x')
plt.show()
fig_2, ax_2 = plt.subplots()
ax_2.plot(np.arange(len(losses)), losses, linewidth = 2)

