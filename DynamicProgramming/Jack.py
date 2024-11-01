import torch as t
import numpy as np
import math 
import matplotlib.pyplot as plt

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

# This tensor gives the probability of different return events, where rows signify from 0 to 40,
# the amount of cars returning from -20 to 20, col 20 is diff = 0
# sum over columns gives the total probability of this event
prob_l1 = t.zeros([21, 41])
prob_l2 = t.zeros([21, 41])
r_sell = t.zeros([21,41])

i = 0
for neg in range(0, 21):
    for pos in range(0, 21):
        prob_l1[neg, pos+i] = poss_func(20 - neg, 3) * poss_func(pos, 3)  
        prob_l2[neg, pos+i] = poss_func(20 - neg, 4) * poss_func(pos, 2)
        r_sell[neg, pos+i] = 10 * min(pos, 20-neg) 
         
    i += 1

def pos_to_value(state):
    return state[0] * 21 + state[1]

def value_to_pos(state):
    return t.reshape(state, (21, 21))
############ Thoughts ################
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
# Wondering whether its p_1*r_1+p_2*r_2 or p_1*p_2*[r_1+r_2]. The second case is formula. The first case is if one sees both events
# as independent and thus the expected value is individual. -> Lets go with second.  
# What happens in loc1 and 2 is independent of each other. Thus we can just sum the cumulative reward

# s' = s + diff -> have to take into account policy 
# Policy Evaluation -> Functional but ugly, can I rewrite it more beautifully ?
def policy_eval(state_value, policy):
    theta = 0.005
    delta = 10 
    while delta > theta:
        time = 0
        for loc1 in range(21):
            for loc2 in range(21):
                time += 1
                curr_state  = t.tensor([loc1, loc2]) 
                value_curr_state = state_value[pos_to_value(curr_state)]
                policy_add  = policy[pos_to_value(curr_state)].item()
                # Loop through all possible future states; col 20 is basically diff = 0 in   
                # prob and reward tensors
                # add policy movements. Positive values indicate cars moving from loc 1 to loc 2
                # negative values loc 2 to loc 1
                # 20 => diff = 0; range of diff is diff - s + policy_change to 20 - s + policy:change
                new_curr = 0
                # Basically goes from future state 0 to 20. -> assume that he moves cars before cars are returned 
                # Do not take into account if more cars are returned or asked for than needed
                for diff_loc1 in range(-loc1 - policy_add,  21 - loc1 - policy_add):
                    for diff_loc2 in range(-loc2 + policy_add,  21 - loc2 + policy_add):
                        #sum of all possibilities for the specific future state f_loc1 and f_loc2 
                        #times the reward of that possibitliy
                        prob_of_nexts_r = prob_l1[:, 20 + diff_loc1] * prob_l2[:, 20 + diff_loc2] 
                        next_reward     = r_sell[:, 20 + diff_loc1] + r_sell[:, 20+ diff_loc2] - policy_add * 2
                        new_curr += t.sum(prob_of_nexts_r * (next_reward + gamma * state_value[pos_to_value(t.tensor([loc1 + diff_loc1, loc2 + diff_loc2]))]))
                state_value[pos_to_value(curr_state)] = new_curr
                delta =  abs(value_curr_state.item() - new_curr.item()) 
                        
def policy_improv(state_value, policy):
    stable = True
    while stable:
        for loc1 in range(21):
            for loc2 in range(21):
                curr_state = t.tensor([loc1, loc2])
                max_state_value = state_value[pos_to_value(curr_state)]
                old_policy = policy[pos_to_value(curr_state)].item()
                # need to limit policy decisions which are sensical
                for new_policy in range(max(-5, -loc2), min(6, loc1)):
                    new_curr = 0
                    for diff_loc1 in range(int(-loc1 - new_policy), max):
                        for diff_loc2 in range(int(-loc2 + new_policy), int(21 - loc2 + new_policy)):
                            try:
                                prob_of_nexts_r = prob_l1[:, 20 + diff_loc1] * prob_l2[:, 20 + diff_loc2]
                            except:
                                print(20, 'Policy:', new_policy, 'Loc1:', loc1,  diff_loc1,'Loc2:', loc2, diff_loc2)
                                print(prob_l1[:, 20 + diff_loc1])
                                print(prob_l2[:, 20 + diff_loc2])
                                exit()
                            next_reward = r_sell[:, 20 + diff_loc1] + r_sell[:, 20 + diff_loc2] - new_policy * 2
                            new_curr += t.sum(prob_of_nexts_r * (next_reward + gamma * state_value[pos_to_value(t.tensor([loc1 + diff_loc1, loc2 + diff_loc2]))]))
                        if new_curr > max_state_value:
                            max_state_value = new_curr
                            if old_policy != new_policy:
                                stable = False
        
 
policy_eval(state_value, policy)
print(state_value)
print(value_to_pos(state_value))
plt.imshow((value_to_pos(state_value)).numpy(), cmap = 'hot', interpolation= 'nearest', origin = 'lower')                 
plt.colorbar()
plt.show()
policy_improv(state_value, policy)
plt.imshow((value_to_pos(policy)).numpy(), cmap = 'hot', interpolation = 'nearest', origin = 'lower')
plt.show()

