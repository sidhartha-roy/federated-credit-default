import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(121, 240)
        self.fc2 = nn.Linear(240, 160)
        self.fc3 = nn.Linear(160, 80)
        self.fc4 = nn.Linear(80, 20)
        self.fc5 = nn.Linear(20, 2)

    def forward(self, x):
        x = x.view(-1, 121)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.log_softmax(self.fc5(x), dim=1)
        return x