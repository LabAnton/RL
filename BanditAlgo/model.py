import torch as t
import math
import kSlotMachine as kS

class Bandit:
    def __init__(self, k: int, epsilon: int, alpha = 0, optimistic = 0, c = 0):
        #k is the number of arms for the bandit
        #epsilon is the number of random explorations
        self.q = t.zeros(k) + optimistic
        self.rounds = t.zeros(k)
        self.epsilon = epsilon
        self.alpha = alpha
        self.k = k
        self.c = c

    def EpsilonGreedy(self):
        p = t.rand(1).item()

        if p < self.epsilon:  
            action = math.floor(t.rand(1).item() * self.k)
        else:
            action = t.argmax(self.q).item() 
        return action 
    
    def UCB(self, t):
        right_term = c * t.sqrt(t.log(t.tensor(t))/self.rounds)
        return t.argmax(self.q + right_term)    
    
    def Update_Average(self, reward: int, action: int):
        self.rounds[action] = self.rounds[action] + 1
        self.q[action] = self.q[action] + 1/self.rounds[action] * (reward - self.q[action])
        
    def Update_WeightedAverage(self, reward: int, action: int):
        self.q[action] = self.q[action] + self.alpha * (reward - self.q[action])
    

