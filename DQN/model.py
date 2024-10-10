import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, env):
        super().__init__()
        
        self.Conv1 = nn.Conv2d(4, 16, 8, stride = 4) 
        self.Pool1 = nn.MaxPool2d(2, stride = 2)
        self.Conv2 = nn.Conv2d(16, 32, 4, stride = 2)
        self.Conv3 = nn.Conv2d(32, 64, 2, stride = 1)
        self.flatten = nn.Flatten()
        self.Linear1 = nn.Linear(768, 512) 
        self.Linear2 = nn.Linear(512, env.action_space.n)

    def forward(self, x):
        x = F.relu(self.Conv1(x))
        x = self.Pool1(x)
        x = F.relu(self.Conv2(x))
        x = self.Pool1(x)
        x = F.relu(self.Conv3(x))
        x = self.flatten(x)
        x = F.relu(self.Linear1(x))
        return self.Linear2(x)
   
