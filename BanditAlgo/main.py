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
moving_mean = 0
epsilons = [1/128, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2]
episodes = 200000

machine = kS.kSlotMachine(k, mean, var, moving_mean)

EG_cum_rewards = t.zeros(len(epsilons), episodes)
EG_rewards = t.zeros(len(epsilons), episodes)

for num, epsilon in enumerate(epsilons):

    EG_agent = model.Bandit(k, epsilon, 0) 

   
    for i in tqdm(range(episodes)):

        EG_action = EG_agent.EpsilonGreedy()    
        EG_reward = machine.Stat_NormDist(EG_action)
        EG_agent.Update_Average(EG_reward, EG_action)
        
        EG_rewards[num, i] = EG_reward
        machine.Update_NormDist() 
    

EG_cum_rewards = t.cumsum(EG_rewards, dim=1) / t.arange(1, episodes+1).repeat(len(epsilons), 1)

EG_avg_reward = t.mean(EG_cum_rewards[:, -100000:], dim =1).reshape(len(epsilons))

plt.plot(epsilons, EG_avg_reward.numpy(), label = f"Epsilon_Greedy")
plt.legend(loc = "lower right")
plt.xscale('log')
plt.show()

