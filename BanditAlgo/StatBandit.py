import kSlotMachine as kS
import model
import matplotlib.pyplot as plt
import torch as t
import numpy as np
from tqdm import tqdm

#Parameters
k = 10 
mean = 2 
var = 1
epsilon = [0, 0.01, 0.1] 
episodes = 2000 
runs = 1000

machine = kS.kSlotMachine(k, mean, var)
print(machine.k)
optimal = t.argmax(machine.k).item()
print(optimal)

runs_cum_reward = t.zeros(len(epsilon), episodes)
runs_cum_optimal = t.zeros(len(epsilon), episodes)

for run in tqdm(range(runs)):
    
    rewards = t.zeros(len(epsilon), episodes) 
    cum_reward = t.zeros(len(epsilon), episodes) 
    optimal_action = t.zeros(len(epsilon), episodes)
    cum_optimal = t.zeros(len(epsilon), episodes)

    for num, epi in enumerate(epsilon):
        agent = model.Bandit(k, epsilon[num])
        for i in range(episodes):

            action = agent.ChooseLever()
            reward = machine.Stat_NormDist(action)    
            agent.UpdateAverage(reward, action) 

            rewards[num, i] = reward
            #cum_reward[num, i] = t.sum(rewards, dim = 1)[num]/i
            if action == optimal:
               optimal_action[num, i] = 1 
            #cum_optimal[num, i] = t.sum(optimal_action, dim = 1)[num]/i 
                 
    cum_reward = t.cumsum(rewards, dim=1) / t.arange(1, episodes+1).repeat(3,1)
    cum_optimal = t.cumsum(optimal_action, dim=1) / t.arange(1, episodes+1)
    runs_cum_reward += cum_reward 
    runs_cum_optimal += cum_optimal

print(runs_cum_optimal)
print(runs_cum_optimal/runs)
color = iter(plt.cm.rainbow(np.linspace(0, 1, 6)))
for index, value in enumerate(epsilon):
    c = next(color)
    plt.plot([*range(episodes)], runs_cum_reward[index, :].numpy()/runs, c=c, label = f"Epsilon: {value}")
plt.legend(loc = "lower right")
plt.show()


for index, value in enumerate(epsilon):
    c = next(color)
    plt.plot([*range(episodes)], runs_cum_optimal[index, :].numpy()/runs, c=c, label = f"Epsilon: {value}")
plt.legend(loc = "lower right")
plt.show()

