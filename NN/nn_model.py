import numpy as np
import torch
import torch.nn as nn


class myCNN(nn.Module):
    """
    CNN architecture
    """

    def __init__(self):
        super().__init__()

        ## layers
        self.inputlayer = nn.Linear(in_features=39*3, out_features=1024)
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=1024, out_features=1024), nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=1024, out_features=1024), nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(in_features=1024, out_features=1024), nn.ReLU()
        )
        self.fc4 = nn.Sequential(
            nn.Linear(in_features=1024, out_features=1024), nn.ReLU()
        )
        self.outputlayer = nn.Linear(in_features=1024, out_features=1)

        ## dropout
        self.dropout = nn.Dropout(0.1)

    # forward propagation
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.inputlayer(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.dropout(x)
        x = self.fc4(x)
        x = self.dropout(x)
        x = self.outputlayer(x)

        return x
