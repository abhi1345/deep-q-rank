import numpy as np
import pandas as pd
import random
from sklearn.utils import shuffle
import torch
import torch.nn as nn
import torch.autograd as autograd
from torchcontrib.optim import SWA
from collections import deque

from preprocess import *

class DQN(nn.Module):
    
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc = nn.Sequential( \
            nn.Linear(self.input_dim[0], 512), \
            nn.ReLU(), \
            nn.Linear(512, 256), \
            nn.ReLU(), \
            nn.Linear(256, self.output_dim))

    def forward(self, state):
        return self.fc(state)
    
class DQNAgent:

    def __init__(self, input_dim, dataset,
                 learning_rate=3e-4, 
                 gamma=0.99,
                 buffer=None,
                 buffer_size=10000, 
                 tau=0.999,
                 swa=False):
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau
        self.model = DQN(input_dim, 1)
        base_opt = torch.optim.Adam(self.model.parameters())
        self.swa = swa
        self.dataset=dataset
        self.MSE_loss = nn.MSELoss()
        self.replay_buffer = buffer
        if swa:
          self.optimizer = SWA(base_opt, swa_start=10, swa_freq=5, swa_lr=0.05)
        else:
          self.optimizer = base_opt

    def get_action(self, state, dataset=None):
        if dataset is None:
            dataset = self.dataset
        inputs = get_multiple_model_inputs(state, state.remaining, dataset)
        model_inputs = autograd.Variable(torch.from_numpy(inputs).float().unsqueeze(0))
        expected_returns = self.model.forward(model_inputs)
        value, index = expected_returns.max(1)
        return state.remaining[index[0]]

    def compute_loss(self, batch, dataset, verbose=False):
        states, actions, rewards, next_states, dones = batch
        model_inputs = np.array([get_model_inputs(states[i], actions[i], dataset)\
            for i in range(len(states))])
        model_inputs = torch.FloatTensor(model_inputs)

        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)

        curr_Q = self.model.forward(model_inputs)
        model_inputs = np.array([get_model_inputs(next_states[i], actions[i], dataset) \
            for i in range(len(next_states))])
        model_inputs = torch.FloatTensor(model_inputs)
        next_Q = self.model.forward(model_inputs)
        max_next_Q = torch.max(next_Q, 1)[0]
        expected_Q = rewards.squeeze(1) + (1 - dones) * self.gamma * max_next_Q

        if verbose:
            print(curr_Q, expected_Q)
        loss = self.MSE_loss(curr_Q.squeeze(0), expected_Q.detach())
        return loss

    def update(self, batch_size, verbose=False):
        batch = self.replay_buffer.sample(batch_size)
        loss = self.compute_loss(batch, self.dataset, verbose)
        train_loss = loss.float()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.swa:
            self.optimizer.swap_swa_sgd()
        return train_loss