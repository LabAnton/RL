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
t.manual_seed(33)
device = t.device('cuda' if t.cuda.is_available() else 'cpu')

reward_per_game = []
replay_memory = []
length_game = []
memory_max = 20000 
games = 400  
epsilon = 1 
minibatch_size = 512 
gamma = 0.9
skip_frame = 4
C = 200

##########Things still to do###########
# 1. Change RGB to Greyscale or increase conv such that last layer is smaller -> Done and added Maxpool2d instead of cropping the image.
# 2. Rewrite train_replay, such that it is better to read and more efficient -> Done 
# 3. Set-up Fixed-Q target -> Done
# 4. Try a training run -> Does not learn after 50 games with epsilon 0.1 and C = 20
# 5. Hyperparameter study
# 6. Get everything on GPU -> Done
# 7. Have not thought about terminal state yet. -> Done
# 8. Introduce gradient clipping to stabilize training -> Not needed since rewards are 1, 0 and -1
# 9. Look at Huber loss for other games
# 10. How are histories implemented ? -> Done
# 11. Look into unsqueeze my implementation seems too complicated.
# 12. Implementation of replay memory is not clean. Terminated states not accounted for and the first ones

env = gym.make("Pong-v4", render_mode = None)

#Network is overparameterized because I get RGB, I have to check later what to do
dqn         = model.DQN(env).to(device)
target_dqn  = model.DQN(env).to(device)
#Initialize such that target_dqn == dqn
target_dqn.load_state_dict(dqn.state_dict())
loss_fn     = nn.MSELoss()
optimizer   = t.optim.SGD(dqn.parameters(), lr = 0.0001, momentum = 0.9)

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
    
def create_history(replay_memory, state):
    if len(replay_memory) >= 8: 
        history_state = t.cat((replay_memory[-1][0][:, -3:], state), dim = 1)
    else:
        #Lets cheat for the beginning and just repeat the same state over all 4 channels. Will not be relevant after the first 4 states. Although training in the beginning will be hard. hmmmm
        history_state = state.repeat(1, 4, 1, 1) 
    
    return history_state

for game in tqdm(range(games)):
    rewards = []
    terminated = False
    state = env.reset()[0]

    state = t.permute(t.unsqueeze(t.tensor(state.astype('float64')), 0), (0, 3, 1, 2)).to(device)
    #Transforming image to greyscale
    state = t.unsqueeze(t.sum(state/3, 1), 0)
    #print('PreState:', state.shape)
    state = create_history(replay_memory, state)
    #print('PreStateHist:', state.shape)
    n = 4 
    while not terminated:
        if n % skip_frame == 0:
            p = t.rand(1).item()
            if p < epsilon: 
                action = env.action_space.sample()
            else:
                action = t.argmax(dqn(state).to(device)).item()

        next_state, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)

        next_state = t.unsqueeze(t.sum(t.permute(t.unsqueeze(t.tensor(next_state.astype('float64')), 0), (0, 3, 1, 2)).to(device)/3, 1), 0)
    #    print('Next_State:', next_state.shape)

        next_state = create_history(replay_memory, next_state)
    #    print('Next_State Hist:' , next_state.shape)
        reward = t.tensor(reward).to(device)
        termi = t.tensor(not terminated).to(device)

        if len(replay_memory) > memory_max:
            replay_memory.pop(0)

        replay_memory.append([state, reward, next_state, termi])
        
        state = next_state

        #Train DQN
        train_replay(replay_memory, minibatch_size)

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
    # Since Pong only has rarely a negative reward, lets delete all 0 rewards state after the run such that it trains more on the negative expiereneces 
    replay_memory = list(filter(lambda x: x == t.tensor(0), replay_memory))

    if epsilon > 0.01:
        epsilon -= 1/1000
     
fig_1, ax_1 = plt.subplots()
ax_1.plot(np.arange(len(reward_per_game)), reward_per_game, 'x')
plt.show()
fig_2, ax_2 = plt.subplots()
ax_2.plot(np.arange(len(length_game)), length_game, 'x')
plt.show()
