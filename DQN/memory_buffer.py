import torch as t
import numpy as np

#What do I need in a Memory buffer?
#1. It needs to save state, reward, next state and terminated state
#2. Has to be size limited due to memory
#3. I need to connect 4 images together and connect the last two frames

class Memory_buffer:
    def __init__(self, memory_max: int, history_length: int, device):
        self.memory_max = memory_max
        self.state = [] 
        self.next_state = []
        self.reward = []
        self.terminated = []
        self.history_length = history_length
        self.device = device
    
    def add(self, state, next_state, reward, terminated):
        #saves states, reward, terminated in list
        if len(state) > self.memory_max:
            self.state.pop(0)
            self.next_state.pop(0)
            self.reward.pop(0)
            self.terminated.pop(0)

        self.state.append(state)
        self.next_state.append(next_state)
        self.reward.append(reward)
        self.terminated.append(terminated)
        
    def sample(self, minibatch_size):
        #Pick minibatch_amount of random numbers
        #Take history of each state
        #Concateate all of the states+historys and return it
        pass
        
    def create_state_hist(self, idx):
        return t.cat(self.state[-4:idx]).to(self.device)

    def create_next_state_hist(self, idx):
        return t.cat(self.state[-4:idx]).to(self.device)
    
    def create_reward_hist(self, idx):
        return t.cat(self.reward[-4:idx]).to(self.device)

    def create_terminated_hist(self,idx):
        return t.cat(self.terminated[-4:idx]).to(self.device)
    
    def last_state(self):
        return state[-1]
