import torch as t
import numpy as np
import math 

# State space is the two location with number of cars in it. Capped at 20
# Reward of 10 for selling a car; reward of -2 for moving a car -> capped at 5 per night
# First location possion distributed requests l = 3 and return l = 3   
# Second location possion distribtued requests l = 4 and return l = 2
# Discount factor of 0.9

#Two spots without cars, State-space has size 21x21 = 441
state_value = t.zeros(441) 
# Policy can go between -5 and 5 depending whether it moves cars from loc 1 to 2 or reverse
policy      = t.zeros(441)
gamma       = 0.9

def poss_func(n, l):
    return (l**n) * math.exp(-l) / math.factorial(n)

# This tensor gives the probability of different return events, where rows signify from 0 to 41,
# the amount of cars returning from -20 to 20 
# sum over columns gives then all possibilities for these events
prob_l1 = t.zeros([21, 41])
prob_l2 = t.zeros([21, 41])
reward_l1 = t.zeros([21,41])
reward_l2 = t.zeros([21, 41])
i = 0
for neg in range(0, 21):
    for pos in range(0, 21):
       prob_l1[neg, pos+i] = poss_func(20 - neg, 3) * poss_func(pos, 3)  
       prob_l2[neg, pos+i] = poss_func(20 - neg, 4) * poss_func(pos, 2)
       reward_l1 = 
    i += 1

def cars_to_pos(state):
    return state[0] * 20 + state[1]

def pick_event_prob(returned_cars):
    pass        
# Prob. of cars at next space is poss_func
# n = s' will loop over all possible future states. However n is equal to 
# n = s + diff  where s is the number of cars right now. 
# positive s are distrbitued through returns while negative through requests
# Problem is we have to sum up over all combinations of future possible states
# f.e. if we have 1 car that is added to place one it could be that 3 cars are
# returned and 2 requested or 4 and 3 and so on...
# assuming return and request are independent, they will be multiplied 
# probabilities for same final event are added to each other
# We can make assumption that every requested car left over will be rewarded with 10
# The amount of cars moved during the night, will have to be taken into account for the future state
# Assume that policy does not influence probabilty of current state
# 

def policy_eval(state, policy):
    theta = 0.005
    while delta > theta:
        delta = 0
        for loc_1 in range(21):
            for loc_2 in range(21):
                curr_state  = t.tensor([loc_1, loc_2]) 
                pos_curr    = cars_to_pos(curr_state)
                v_curr      = state_value[pos_curr_state]
                #Loop through all possible future states  
                for i in range(442):    
                    pass
                    #Calculate specific future state to go to 
                
                



