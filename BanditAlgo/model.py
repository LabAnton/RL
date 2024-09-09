import torch as t
import math
import kSlotMachine as kS

class Bandit:
    def __init__(self, k: int, epsilon = 0 , alpha = 0, optimistic = 0, c = 0):
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
    
    def UCB(self, time):
        #can still be written more beautifully
        if not self.rounds.all():
            sub = self.rounds.clone()
            sub[sub == 0] = 1
            right_term = self.c * t.sqrt(t.log(t.tensor(time))/sub)
        else:
            right_term = self.c * t.sqrt(t.log(t.tensor(time))/self.rounds)
        return t.argmax(self.q + right_term).item()    
    
    def Update_Average(self, reward: int, action: int):
        self.rounds[action] = self.rounds[action] + 1
        self.q[action] = self.q[action] + 1/self.rounds[action] * (reward - self.q[action])
        
    def Update_WeightedAverage(self, reward: int, action: int):
        self.q[action] = self.q[action] + self.alpha * (reward - self.q[action])
    
    def Gradient_Pick(self):
        p = t.rand(1).item()
        pi_t = t.cumsum(t.exp(self.q)/t.sum(t.exp(self.q)), dim = 0)
        #print(pi_t, p)
        #print(self.k,  t.sum(t.where(p > pi_t, 0, 1)).item(), self.k - t.sum(t.where(p < pi_t, 1, 0)).item())
        return self.k - t.sum(t.where(p > pi_t, 0, 1)).item()  

    def Update_Gradient(self, reward: int, action: int, R_t: float):
        pi_t = t.exp(self.q)/t.sum(t.exp(self.q))

        self.q = self.q - self.alpha * (reward - R_t) *(pi_t)
        #print('Q-table', self.q)
        #print(pi_t)
        self.q[action] += self.alpha * (reward - R_t) 
        
