import gymnasium as gym
import math
import time
import torch as t
import matplotlib.pyplot as plt
from tqdm import tqdm

t.manual_seed(333)

env = gym.make("Taxi-v3", render_mode = "human")
env.reset()

#Parameters
alpha = 0.7 #learning rate
gamma = 0.618 #discount factor
epsilon = 1
max_epsilon = 1
min_epsilon = 0.01
decay = 0.01

train_episodes = 500
test_episodes = 100
max_steps = 199

#Initiliaze state-action tensor
Q = t.zeros([env.observation_space.n, env.action_space.n])

#keep track of rewards per run and how much it explores
training_rewards = []
exploration_steps = []

for episode in tqdm(range(train_episodes)):
    #reset the environment, get only state and ignore action mask and prob
    state = env.reset()[0]
    total_rewards = 0
    total_exploration = 0
    
    #number of tries for one episode
    for step in range(max_steps):
        prob = t.rand(1).item() 
        
        #If the prob is higher than our epsilon than we will exploit. In the beginning since epsilone is 1, we will alwazs explore with a random action. 
        if prob > epsilon:
            #Pick action value in current state with the highest value. In our intilized state-action table this would be "Move south (down)"
            action = t.argmax(Q[state,:]).item()

        else:
            total_exploration += 1
            action = env.action_space.sample()
        #observation is basically the next env state, rewards is the rewards of this action which are predefined for the env
        observation, reward, terminated, truncated, info = env.step(action)
        #Bellman equation to update the reward of the state-action pair; the current state is weighed against the reward which could be accumulated in the next state, alpha and gamma are the factors for that 
        try:
            Q[state, action] = Q[state, action] + alpha * (reward + gamma * t.max(Q[observation, :]) - Q[state, action]) 
        except ValueError:
            print("Action: ", action)
        
        total_rewards += reward 
        state = observation

        if terminated == True:
            break
    
    #Decreasing the amount of exploration done per episode
    epsilon = min_epsilon + (max_epsilon-min_epsilon) * math.exp(-decay*episode)
    
    training_rewards.append(total_rewards)
    exploration_steps.append(total_exploration)

env.close()
x = range(train_episodes)
plt.plot(x, training_rewards)
plt.xlabel("Episode")
plt.ylabel("Training total reward")
plt.show()
plt.plot(x, exploration_steps)
plt.xlabel("Episode")
plt.ylabel("Number of exploration steps")
plt.title("Exploration in each episode")
plt.show()
env.close()
print(epsilon)
t.save(Q, "Action-StatePair_1.pt")
