import kSlotMachine as kS
import model
import matplotlib.pyplot as plt
import torch as t
import numpy as np

#Parameters
k = 10 
mean = 2 
var = 1.5
epsilon = [0, 0.01, 0.1] 
episodes = 2000

machine = kS.kSlotMachine(k, mean, var)
print(machine.k)
optimal = t.argmax(machine.k).item()

rewards = t.zeros(len(epsilon), episodes) 
cum_reward = t.zeros(len(epsilon), episodes) 
optimal_action = t.zeros(len(epsilon), episodes)
cum_optimal = t.zeros(len(epsilon), episodes)

for num, epi in enumerate(epsilon):
    agent = model.Bandit(k, epsilon[num])
    for i in range(episodes):

        action = agent.ChooseLever()
        reward = machine.Pick_NormDist(action)    
        agent.UpdateAverage(reward, action) 

        rewards[num, i] = reward
        cum_reward[num, i] = t.sum(rewards, dim = 1)[num]/i
        if action == optimal:
           optimal_action[num, i] = 1 
        cum_optimal[num, i] = t.sum(optimal_action, dim = 1)[num]/i 
             

color = iter(plt.cm.rainbow(np.linspace(0, 1, 3)))
for index, value in enumerate(epsilon):
    c = next(color)
    plt.plot([*range(episodes)], cum_reward[index, :].numpy(), c=c, label = f"Epsilon: {value}")
plt.legend(loc = "lower right")
plt.show()

