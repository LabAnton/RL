import kSlotMachine as kS
import model
import matplotlib.pyplot as plt
import torch as t
import numpy as np
from tqdm import tqdm

#Parameters
k = 10 
mean = 2 
var = 2 
constants = [1] 
episodes = 2000
runs = 50 

machine = kS.kSlotMachine(k, mean, var)
print(machine.k)
optimal = t.argmax(machine.k).item()
print('Optimal:', optimal)

runs_cum_reward = t.zeros(len(constants), episodes)
runs_cum_optimal = t.zeros(len(constants), episodes)

for run in tqdm(range(runs)):
    
    rewards = t.zeros(len(constants), episodes) 
    cum_reward = t.zeros(len(constants), episodes) 
    optimal_action = t.zeros(len(constants), episodes)
    action_taken = t.zeros(len(constants), episodes)
    cum_optimal = t.zeros(len(constants), episodes)

    for num, c in enumerate(constants):
        agent = model.Bandit(k, c = c)
        for i in range(episodes):

            action = agent.UCB(i+1)
            reward = machine.Stat_NormDist(action)    
            agent.Update_Average(reward, action) 

            rewards[num, i] = reward
            #cum_reward[num, i] = t.sum(rewards, dim = 1)[num]/i
            action_taken[num,i] = action
            if action == optimal:
               optimal_action[num, i] = 1 
            #cum_optimal[num, i] = t.sum(optimal_action, dim = 1)[num]/i 
                 
    cum_reward = t.cumsum(rewards, dim=1) / t.arange(1, episodes+1).repeat(len(constants) ,1)
    cum_optimal = t.cumsum(optimal_action, dim=1) / t.arange(1, episodes+1)
    runs_cum_reward += cum_reward 
    runs_cum_optimal += cum_optimal

print(agent.q)
color = iter(plt.cm.rainbow(np.linspace(0, 1, 2*len(constants))))
for index, value in enumerate(constants):
    c = next(color)
    plt.plot([*range(episodes)], runs_cum_reward[index, :].numpy()/runs, c=c, label = f"Constant: {value}")
plt.legend(loc = "lower right")
plt.show()

for index, value in enumerate(constants):
    c = next(color)
    plt.plot([*range(episodes)], runs_cum_optimal[index, :].numpy()/runs, c=c, label = f"Constant: {value}")
plt.legend(loc = "lower right")
plt.show()

