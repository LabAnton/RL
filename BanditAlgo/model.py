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
    
    def UCB(self, t):
        right_term = c * t.sqrt(t.log(t.tensor(t))/self.rounds)
        print(right_term.shape)
        return t.argmax(self.q + right_term).item()    
    
    def Update_Average(self, reward: int, action: int):
        self.rounds[action] = self.rounds[action] + 1
        self.q[action] = self.q[action] + 1/self.rounds[action] * (reward - self.q[action])
        
    def Update_WeightedAverage(self, reward: int, action: int):
        self.q[action] = self.q[action] + self.alpha * (reward - self.q[action])
    
    def Gradient_Pick(self):
        p = t.rand(1).item()
        # Has to some to one
        pi_t = t.cumsum(t.exp(self.q)/t.sum(t.exp(self.q)), dim = 0)
        #somehow pick p from intervals in prob  
        for index, prob in enumerate(pi_t):
            try:
                if p < prob.item():
                    return index
            except:
                print(self.q, pi_t)
                exit()
        return self.k - 1

    def Update_Gradient(self, reward: int, action: int, R_t: float):
        pi_t = t.exp(self.q)/t.sum(t.exp(self.q))
        #print('Distribution: ', pi_t, 'R_t:', R_t)

        #print('Before: ', self.q)
        self.q = self.q - self.alpha * (reward - R_t) *(pi_t)
        self.q[action] += self.alpha * (reward - R_t) 
        #print('Updates:' , self.q)
        
