import numpy as np
import torch
import torch.nn as nn


class myCNN(nn.Module):
    """
    CNN architecture
    """

    def __init__(self):
        super().__init__()

        # input size: 64^3
        ## convolution
        self.layer1 = nn.Sequential(
            nn.Conv3d(
                in_channels=1,
                out_channels=2,
                kernel_size=3,
                stride=1,
                padding=0,
            ),
            ##
            nn.ReLU(),
            ##
            # size: 2 * 62^3
            nn.BatchNorm3d(num_features=2),
            nn.MaxPool3d(kernel_size=2),
            nn.ReLU(),
            # size: 2 * 31^3
        )

        self.layer2 = nn.Sequential(
            nn.Conv3d(
                in_channels=2,
                out_channels=128,
                kernel_size=2,
                stride=1,
                padding=0,
            ),
            ##
            nn.ReLU(),
            ##
            # size: 64 * 30^3
            nn.BatchNorm3d(num_features=128),
            nn.MaxPool3d(kernel_size=2),
            # size: 64 * 15^3
            nn.ReLU(),
        )

        self.layer3 = nn.Sequential(
            nn.Conv3d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=0,
            ),
            nn.ReLU(),
            # size: 64 * 7^3
            nn.Conv3d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=0,
            ),
            nn.ReLU(),
            # size: 64 * 5^3
            nn.Conv3d(
                in_channels=128,
                out_channels=128,
                kernel_size=2,
                stride=1,
                padding=0,
            ),
            ##
            nn.ReLU(),
            ##
            # size: 128 * 4^3
            nn.BatchNorm3d(num_features=128),
            nn.ReLU(),
        )

        ### need flaten here

        ## Fully Connected
        self.layer4 = nn.Sequential(
            nn.Linear(in_features=int(128 * 4 ** 3), out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=1)
        )

    # forward propagation
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(-1, int(128 * 4 ** 3))  # flatten
        x = self.layer4(x)

        return x
