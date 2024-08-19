import torch as t

class kSlotMachine:
    def __init__(self, k: int, mean:int, var: int):
        # k is the number of arms or choices
        # mean is the highest and lowest value the distribution can have 
        # var is the variance of the distirbution
        self.k = -2*mean * t.rand(k) + mean
        self.var = t.rand(k)

    def Pick_NormDist(self, *arm: int):
        return t.normal(self.k[arm], self.var[arm])

