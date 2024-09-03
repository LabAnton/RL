import kSlotMachine as kS
import model
import matplotlib.pyplot as plt
import torch as t
import numpy as np
from tqdm import tqdm

#Parameters
k = 10
mean = 2 
var = 3 
alphas = [1/8, 1/4, 1/2, 1, 2, 4] 
episodes = 10000 
moving_mean = 0
runs = 40 

runs_optimal_acts = t.zeros(len(alphas),episodes)
for n, alpha in enumerate(alphas):

    for run in tqdm(range(runs)):

        R_t = t.zeros(episodes)
        R_t[0] = 1
        machine = kS.kSlotMachine(k, mean, var, moving_mean)
        grad_bandit = model.Bandit(k, alpha = alpha)
        optimal_act = t.zeros(episodes)
        
        for epi in range(episodes):

            optimal = t.argmax(machine.k).item()
            action = grad_bandit.Gradient_Pick()
            reward = machine.Stat_NormDist(action)
            
            grad_bandit.Update_Gradient(reward, action, t.sum(R_t) / (epi+1))
            R_t[epi] = reward

            if action == optimal:
                optimal_act[epi] = 1
                
        runs_optimal_acts[n] += t.cumsum(optimal_act, dim=0) / t.arange(1, episodes+1) 


color = iter(plt.cm.rainbow(np.linspace(0, 1, len(alphas))))
for i in range(len(alphas)):
    c = next(color)
    plt.plot([*range(episodes)], runs_optimal_acts[i].numpy() / runs, c=c, label = f'{alphas[i]}')
plt.legend(loc = "lower right")
plt.show()

