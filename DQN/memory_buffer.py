import torch as t
import numpy as np

#What do I need in a Memory buffer?
#1. It needs to save state, reward, next state and terminated state
#2. Has to be size limited due to memory
#3. I need to connect 4 images together and connect the last two frames

class Memory_buffer:
    def __init__(self, max_memory: int, history_length: int, minibatch_size: int):
        self.state = [] 
        self.next_state = []
        self.reward = []
        self.terminated = []
        self.max_memory = max_memory 
        self.histolen = history_length
        self.minibatch_size = minibatch_size
    
    def add(self, state, next_state, reward, terminated):
        #saves states, reward, terminated in list
        if len(state) > self.max_memory:
            self.state.pop(0)
            self.next_state.pop(0)
            self.reward.pop(0)
            self.terminated.pop(0)

        assert len(state.shape) == 4
        assert len(next_state.shape) == 4
        assert t.is_tensor(reward)
        assert t.is_tensor(terminated)  

        #In case I miscalculated memory usage
        try:
            self.state.append(state)
            self.next_state.append(next_state)
            self.reward.append(reward)
            self.terminated.append(terminated)
        except:
            print(len(state))
            self.max_memory = len(state)-10
        
    def sample(self):
        #we are looping a lot here with the histories. Can I save it more efficiently ?
        random_choice   = np.random.choice(np.arange(self.histolen, len(self.state)), self.minibatch_size)
        states          = t.cat([self.create_state_hist(i) for i in random_choice], dim = 0)
        next_states     = t.cat([self.create_next_state_hist(i) for i in random_choice], dim = 0) 
        rewards         = t.stack([self.reward[i] for i in random_choice])
        terminated      = t.stack([self.terminated[i] for i in random_choice]) 
        
        return states, next_states, rewards, terminated
        
    def create_state_hist(self, idx):
        return t.cat([self.state[j] for j in range(idx-self.histolen, idx)], dim = 1)

    def create_next_state_hist(self, idx):
        return t.cat([self.next_state[j] for j in range(idx-self.histolen, idx)], dim = 1)
    
    def last_state(self):
        return state[-1]
    
    def length(self):
       return len(self.state) 
