import torch as t
import math
import kSlotMachine as kS

class Bandit:
    def __init__(self, k: int, epsilon: int):
        #k is the number of arms for the bandit
        #epsilon is the number of random explorations
        self.q = t.zeros(k)
        self.rounds = t.zeros(k)
        self.epsilon = epsilon

    def ChooseLever(self):
        p = t.rand(1).item()

        if p < self.epsilon:  
            action = math.floor(t.rand(1).item() * 10)
        else:
            action = t.argmax(self.q).item() 
        return action 
    
    def UpdateAverage(self, reward: int, action: int):
        self.rounds[action] = self.rounds[action] + 1
        self.q[action] = self.q[action] + 1/self.rounds[action] * (reward - self.q[action])
        #print(self.q)
        

