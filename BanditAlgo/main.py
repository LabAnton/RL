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
moving_mean = 0
epsilons = [1/124, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2]
constants = [1/124, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1, 2]
alphas = [1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1, 2]
episodes = 200000 
runs = 20

#Epsilon-Greedy Algorithm
EG_run_avg_rewards = t.zeros(len(epsilons))
EG_cum_rewards = t.zeros(len(epsilons), episodes)
EG_rewards = t.zeros(len(epsilons), episodes)

#Upper-Confidence-Bound Action UCB_run_avg_rewards = t.zeros(len(constants))
UCB_run_avg_rewards =t.zeros(len(constants))
UCB_cum_rewards = t.zeros(len(constants), episodes)
UCB_rewards = t.zeros(len(constants), episodes)

#Gradient Bandit
GB_run_avg_rewards = t.zeros(len(alphas))
GB_cum_rewards = t.zeros(len(alphas), episodes)


for run in range(runs):
    print('RUN NUMBER: ', run+1)
    machine = kS.kSlotMachine(k, mean, var, moving_mean)

    GB_rewards = t.zeros(len(alphas), episodes)
    for num, alpha in enumerate(alphas):

        GB_agent = model.Bandit(k, alpha = alpha) 

        for i in tqdm(range(episodes)):

            GB_action = GB_agent.Gradient_Pick()    
            GB_reward = machine.Stat_NormDist(GB_action)
            GB_agent.Update_Gradient(GB_reward, GB_action, R_t = t.sum(GB_rewards[num, :]).item()/(i+1))
        
            GB_rewards[num, i] = GB_reward
            
    
    # Take cumulutative sum over episodes and divide my the episode
    GB_cum_rewards = t.cumsum(GB_rewards, dim=1) / t.arange(1, episodes+1).repeat(len(alphas), 1)
    # Take the average 100000 data points over all epsilon and reshape it into a [len(epsilon)] shape
    GB_avg_reward = t.mean(GB_cum_rewards[:, -100000:], dim =1).reshape(len(alphas))
    print(GB_avg_reward)
    # Add to the average run reward, every position is one epsilon 
    GB_run_avg_rewards += GB_avg_reward    
    print(GB_run_avg_rewards)


    for num, epsilon in enumerate(epsilons):

        EG_agent = model.Bandit(k, epsilon = epsilon) 

        for i in tqdm(range(episodes)):

            EG_action = EG_agent.EpsilonGreedy()    
            EG_reward = machine.Stat_NormDist(EG_action)
            EG_agent.Update_Average(EG_reward, EG_action)
            
            EG_rewards[num, i] = EG_reward
        
    # Take cumulutative sum over episodes and divide by the episode
    EG_cum_rewards = t.cumsum(EG_rewards, dim=1) / t.arange(1, episodes+1).repeat(len(epsilons), 1)
    # Take the average 100000 data points over all epsilon and reshape it into a [len(epsilon)] shape
    EG_avg_reward = t.mean(EG_cum_rewards[:, -100000:], dim =1).reshape(len(epsilons))
    # Add to the average run reward, every position is one epsilon 
    EG_run_avg_rewards += EG_avg_reward    

    for num, c in enumerate(constants):

        UCB_agent = model.Bandit(k, c = c)
       
        for i in tqdm(range(episodes)):

             UCB_action = UCB_agent.UCB(i+1)    
             UCB_reward = machine.Stat_NormDist(UCB_action)
             UCB_agent.Update_Average(UCB_reward, UCB_action)
           
             UCB_rewards[num, i] = UCB_reward
       
    # Take cumulutative sum over episodes and divide my the episode
    UCB_cum_rewards = t.cumsum(UCB_rewards, dim=1) / t.arange(1, episodes+1).repeat(len(constants), 1)
    # Take the average 100000 data points over all epsilon and reshape it into a [len(epsilon)] shape
    UCB_avg_reward = t.mean(UCB_cum_rewards[:, -100000:], dim =1).reshape(len(constants))
    # Add to the average run reward, every position is one epsilon 
    UCB_run_avg_rewards += UCB_avg_reward 

# Divide by number of runs to get the average reward
EG_run_avg_rewards = EG_run_avg_rewards/runs
UCB_run_avg_rewards = UCB_run_avg_rewards/runs
GB_run_avg_rewards = GB_run_avg_rewards/runs

plt.plot(epsilons, EG_run_avg_rewards.numpy(), label = f"Epsilon_Greedy")
plt.plot(constants, UCB_run_avg_rewards.numpy(), label = 'UpperBoundAction')
plt.plot(alphas, GB_run_avg_rewards.numpy(), label = 'GradientBandit')
plt.legend(loc = "lower right")
plt.title(f'Parameter study; Norm distribution with mean in: {t.argmax(machine.k)}')
plt.xscale('log')
plt.ylabel("Average Reward over last 100.000 Steps")
plt.xlabel("Parameters")
plt.show()

