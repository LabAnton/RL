import kSlotMachine as kS
import model
import matplotlib.pyplot as plt
import torch as t
import numpy as np
from tqdm import tqdm

#Parameters
k = 10
mean = 0 
var = 2 
epsilon = 0.1 
episodes = 40000
alpha = 0.1 
moving_mean = 0
runs = 150 

machine = kS.kSlotMachine(k, mean, var, moving_mean)

runs_cum_reward_avg = t.zeros(episodes)
runs_cum_optimal_avg = t.zeros(episodes)
runs_cum_reward_weighted = t.zeros(episodes)
runs_cum_optimal_weighted = t.zeros(episodes)

for run in tqdm(range(runs)):
    rewards_avg = t.zeros(episodes)
    rewards_weighted = t.zeros(episodes)

    optimal_avg = t.zeros(episodes)
    optimal_weighted = t.zeros(episodes)

    agent_avg = model.Bandit(k, epsilon,  0)
    agent_weighted = model.Bandit(k, epsilon, alpha)

    for epi in range(episodes):
        optimal = t.argmax(machine.k).item()
        action_avg = agent_avg.EpsilonGreedy()
        action_weighted = agent_weighted.EpsilonGreedy()

        reward_avg = machine.Stat_NormDist(action_avg)
        reward_weighted = machine.Stat_NormDist(action_weighted)

        agent_avg.Update_Average(reward_avg, action_avg)
        agent_weighted.Update_WeightedAverage(reward_weighted, action_weighted)

        machine.Update_NormDist()

        rewards_avg[epi] = reward_avg
        rewards_weighted[epi] = reward_weighted 

        if action_avg == optimal:
            optimal_avg[epi] = 1
        if action_weighted == optimal:
            optimal_weighted[epi] = 1
            
    cum_reward_avg = t.cumsum(rewards_avg, dim=0) / t.arange(1, episodes+1)
    cum_optimal_avg = t.cumsum(optimal_avg, dim=0) / t.arange(1, episodes+1)

    cum_reward_weighted = t.cumsum(rewards_weighted, dim=0) / t.arange(1, episodes+1)
    cum_optimal_weighted = t.cumsum(optimal_weighted, dim=0) / t.arange(1, episodes+1)
     
    runs_cum_reward_avg += cum_reward_avg
    runs_cum_optimal_avg += cum_optimal_avg
    runs_cum_reward_weighted += cum_reward_weighted
    runs_cum_optimal_weighted += cum_optimal_weighted 
    

color = iter(plt.cm.rainbow(np.linspace(0, 1, 4)))
c = next(color)
plt.plot([*range(episodes)], runs_cum_reward_avg.numpy() / runs, c=c, label = "Avg")
c = next(color)
plt.plot([*range(episodes)], runs_cum_reward_weighted.numpy() / runs, c=c, label = "Weighted")
plt.legend(loc = "lower right")
plt.show()

c = next(color)
plt.plot([*range(episodes)], runs_cum_optimal_avg.numpy() / runs, c=c, label = "Avg")
c = next(color)
plt.plot([*range(episodes)], runs_cum_optimal_weighted.numpy() / runs, c=c, label = "Weighted")
plt.legend(loc = "lower right")
plt.show()




