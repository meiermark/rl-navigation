#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(DQN, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class Dueling_DQN(nn.Module):
    def __init__(self, state_size, action_size, seed):
        super(Dueling_DQN, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.action_size = action_size

        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)

        self.fc1_adv = nn.Linear(in_features=64, out_features=64)
        self.fc2_adv = nn.Linear(in_features=64, out_features=action_size)

        self.fc1_val = nn.Linear(in_features=64, out_features=64)
        self.fc2_val = nn.Linear(in_features=64, out_features=1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        adv = F.relu(self.fc1_adv(x))
        adv = self.fc2_adv(adv)

        val = F.relu(self.fc1_val(x))
        val = self.fc2_val(val).expand(x.size(0), self.action_size)

        x = val + adv - adv.mean(1).unsqueeze(1).expand(x.size(0), self.action_size)
        return x

