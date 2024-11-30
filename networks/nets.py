"""
This file contains all the networks created in this project for making the comparison
---------------ALERT!!!!!: if you want to train the model based on initialization in the notebook files
"""
import random

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F



class DQN(nn.Module):
    """
    Deep Q Network class
    """

    def __init__(self, observation_space, action_space):
        """
        set up neural nets
        """
        super().__init__()

        # neurons per hidden  layer = 2 * observations space => 660
        neurons_per_layer = observation_space

        # set starting exploration rate to zero because we do inference
        self.exploration_rate = 0

        # set un action space
        self.action_space = action_space
        self.obs_space = observation_space

        # set up the device for make  calculations
        # CPU should be faster for this case wit GPU
        self.device = 'cpu'
        self.net = nn.Sequential(
            nn.Linear(observation_space, neurons_per_layer).to(self.device),
            nn.ReLU(),
            nn.Linear(neurons_per_layer, neurons_per_layer).to(self.device),
            nn.ReLU(),
            nn.Linear(neurons_per_layer, neurons_per_layer).to(self.device),
            nn.ReLU(),
            nn.Linear(neurons_per_layer, action_space).to(self.device)
        ).to(self.device)

    def act(self, state):
        """
        Act either randomly or by redicting action that gives max Q
        """

        # Convert to Tensor
        # reshape state into 2D array with obs as first 'row'
        state = torch.tensor(np.reshape(state, [1, self.obs_space]), dtype=torch.float32)

        # Act randomly if random number < exploration rate
        if np.random.rand() < self.exploration_rate:
            action = random.randrange(self.action_space)
        else:
            with torch.no_grad():
                # Otherwise get predicted Q values of actions
                q_values = self.net(state)
                # get index of action with best Q
                action = np.argmax(q_values.detach().numpy()[0])

        # we return also for run the system in the same function of A2C
        return action

    def forward(self, x):
        """Forward pass and return the action values """
        x = x.to(self.device)
        return self.net(x)


class Actor(nn.Module):
    """Actor Network"""

    def __init__(self, observation_space, action_space):
        """ set up the act net"""
        super().__init__()
        self.float()
        # neurons per hidden  layer
        neurons_per_layer = observation_space
        self.device = 'cpu'
        self.net = nn.Sequential(
            nn.Linear(observation_space, neurons_per_layer).to(self.device),
            nn.ReLU().to(self.device),
            nn.Linear(neurons_per_layer, action_space).to(self.device)
            # we output the distribution over action space(before softmax)
        ).to(self.device)

    def act(self, state):
        """Act  and return action ( ONLY for inference)"""

        # convert to Tensor
        state = torch.tensor(state, dtype=torch.float32, device=self.device)

        probs = F.softmax(self.net(state), dim=-1)

        # Create a categorical distribution over actions
        action_dist = torch.distributions.Categorical(probs)

        # sample an action
        action = action_dist.sample().item()
        action_prob = probs[action].item()

        return action

    def forward(self, x):
        """ Forward pass that return the logits before softmax """
        return self.net(x.to(self.device))
