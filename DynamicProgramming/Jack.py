import torch as t
import numpy as np
import math 

# State space is the two location with number of cars in it. Capped at 20
# Reward of 10 for selling a car; reward of -2 for moving a car -> capped at 5 per night
# First location possion distributed requests l = 3 and return l = 3   
# Second location possion distribtued requests l = 4 and return l = 2
# Discount factor of 0.9

#Two spots without cars, State-space has size 21x21 = 441
state_value = t.rand(441) 
# Policy can go between -5 and 5 depending whether it moves cars from loc 1 to 2 or reverse
policy      = t.zeros(441)
gamma       = 0.9

def state_to_policy(state):
    return state[0] * 20 + state[1]

def poss_func(n, l):
    return (l**n) * math.exp(-l) / math.factorial(n)
    
def policy_eval(state, policy):
    theta = 0.005
    while delta > theta:
        delta = 0
        for loc_1 in range(21):
            for loc_2 in range(21):
                v       = t.tensor([loc_1, loc_2]) 
                # Prob. of cars at next space is poss_func
                # n = s' will loop over all possible future states. However n is equal to 
                # n = s + diff  where s is the number of cars right now. 
                # positive s are distrbitued through returns while negative through requests
                new_v   = prob(loc_1, 3) * prob( 




