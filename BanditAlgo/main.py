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
epsilons = [1/128, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2]
constants = [1/16, 1/8, 1/4, 1/2, 1, 2, 4]
alphas = [1/4, 1/2, 1, 2]
episodes = 200000 
runs = 50 

#Epsilon-Greedy Algorithm
EG_run_avg_rewards = t.zeros(len(epsilons))
EG_cum_rewards = t.zeros(len(epsilons), episodes)
EG_rewards = t.zeros(len(epsilons), episodes)

#Upper-Confidence-Bound Action
UCB_run_avg_rewards = t.zeros(len(constants))
UCB_cum_rewards = t.zeros(len(constants), episodes)
UCB_rewards = t.zeros(len(constants), episodes)

#Gradient Bandit
GB_run_avg_rewards = t.zeros(len(alphas))
GB_cum_rewards = t.zeros(len(alphas), episodes)
GB_rewards = t.zeros(len(alphas), episodes)


for run in range(runs):

    for num, alpha in enumerate(alphas):

        machine_GB = kS.kSlotMachine(k, mean, var, moving_mean)
        GB_agent = model.Bandit(k, alpha = alpha) 

        for i in tqdm(range(episodes)):

            GB_action = GB_agent.Gradient_Pick()    
            GB_reward = machine_GB.Stat_NormDist(GB_action)
            #print('Optimal:', optimal, 'Action:', GB_action, 'Agent-Optimal:', t.argmax(GB_agent.q).item())
            #print('Reward:', GB_reward, ' R_t:', (t.sum(GB_rewards[num, :i]/i)).item())

            GB_agent.Update_Gradient(GB_reward, GB_action, R_t = (t.sum(GB_rewards[num, :i]/i)).item())
            
            try:    
                GB_rewards[num, i] = GB_reward
            except:
                print(GB_action)
                print(GB_rewards.shape, GB_reward)
                print(GB_rewards)
                exit()
        
    
    # Take cumulutative sum over episodes and divide my the episode
    GB_cum_rewards = t.cumsum(GB_rewards, dim=1) / t.arange(1, episodes+1).repeat(len(alphas), 1)
    # Take the average 100000 data points over all epsilon and reshape it into a [len(epsilon)] shape
    GB_avg_reward = t.mean(GB_cum_rewards[:, -100000:], dim =1).reshape(len(alphas))
    # Add to the average run reward, every position is one epsilon 
    GB_run_avg_rewards += GB_avg_reward    


    for num, epsilon in enumerate(epsilons):

        machine_EG = kS.kSlotMachine(k, mean, var, moving_mean)
        EG_agent = model.Bandit(k, epsilon = epsilon, alpha = 0.1) 

        for i in tqdm(range(episodes)):

            EG_action = EG_agent.EpsilonGreedy()    
            EG_reward = machine_EG.Stat_NormDist(EG_action)
            EG_agent.Update_Weighted_Average(EG_reward, EG_action)
            
            EG_rewards[num, i] = EG_reward
            machine_EG.Update_NormDist() 
        
    # Take cumulutative sum over episodes and divide my the episode
    EG_cum_rewards = t.cumsum(EG_rewards, dim=1) / t.arange(1, episodes+1).repeat(len(epsilons), 1)
    # Take the average 100000 data points over all epsilon and reshape it into a [len(epsilon)] shape
    EG_avg_reward = t.mean(EG_cum_rewards[:, -100000:], dim =1).reshape(len(epsilons))
    # Add to the average run reward, every position is one epsilon 
    EG_run_avg_rewards += EG_avg_reward    

    for num, c in enumerate(constants):

        machine_UCB = kS.kSlotMachine(k, mean, var, moving_mean)
        UCB_agent = model.Bandit(k, mean, alpha = 0.1, c = c)
        
        for i in tqdm(range(episodes)):

            UCB_action = UCB_agent.UCB()    
            UCB_reward = machine_UCB.Stat_NormDist(EG_action)
            UCB_agent.Update_Average(UCB_reward, UCB_action)
            
            UCB_rewards[num, i] = UCB_reward
            machine_UCB.Update_NormDist() 
        
    # Take cumulutative sum over episodes and divide my the episode
    UCB_cum_rewards = t.cumsum(UCB_rewards, dim=1) / t.arange(1, episodes+1).repeat(len(constants), 1)
    # Take the average 100000 data points over all epsilon and reshape it into a [len(epsilon)] shape
    UCB_avg_reward = t.mean(UCB_cum_rewards[:, -100000:], dim =1).reshape(len(constants))
    # Add to the average run reward, every position is one epsilon 
    UCB_run_avg_rewards += UCB_avg_reward 
    print(run)

# Divide by number of runs to get the average reward
EG_run_avg_rewards = EG_run_avg_rewards/runs
UCB_run_avg_rewards = UCB_run_avg_rewards/runs

plt.plot(epsilons, EG_run_avg_rewards.numpy(), label = f"Epsilon_Greedy")
plt.plot(constants, UCB_run_avg_rewards.numpy(), label = 'UpperBoundAction')
plt.legend(loc = "lower right")
plt.xscale('log')
plt.ylabel("Average Reward over last 100000 Steps")
plt.xlabel("Parameters")
plt.show()

