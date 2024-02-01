import numpy as np
import pandas as pd
import random
from sklearn.utils import shuffle
import torch
import torch.nn as nn
import torch.autograd as autograd
from torchcontrib.optim import SWA
from collections import deque

from util.preprocess import *

class DQN(nn.Module):
    
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        # self.fc = nn.Sequential( \
        #     nn.Linear(self.input_dim[0], 32), \
        #     nn.ReLU(), \
        #     nn.Linear(32, self.output_dim))

        self.fc = nn.Sequential(
        nn.Linear(self.input_dim[0], 384),  # Input layer to first hidden layer
        nn.ReLU(),
        nn.Linear(384, 192),  # First hidden layer to second hidden layer
        # nn.ReLU(),
        # nn.Linear(192, 96),  # Second hidden layer to third hidden layer
        # nn.ReLU(),
        # nn.Linear(96, 48),  # Third hidden layer to fourth hidden layer
        # nn.ReLU(),
        nn.Linear(192, 24),  # Fourth hidden layer to fifth hidden layer
        nn.ReLU(),
        nn.Linear(24, self.output_dim)  # Fifth hidden layer to output layer
        )

    def forward(self, state):
        # print(f"len(state): { len(state)}")
        return self.fc(state)
    
class DQNAgent:

    def __init__(self, input_dim, dataset,
                 learning_rate=1e-4, 
                 gamma=1,
                 buffer=None,
                 buffer_size=10000, 
                 tau=0.99,
                 swa=False,
                 pre_trained_model=None):

        # print(f"input_dim:{input_dim}")
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau
        self.model = DQN(input_dim, 1)
        if pre_trained_model:
            self.model = pre_trained_model
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
        # print("get_action")
        if dataset is None:
            dataset = self.dataset
        inputs = get_multiple_model_inputs(state, state.remaining, dataset)
        
        model_inputs = autograd.Variable(torch.from_numpy(inputs).float().unsqueeze(0))
        expected_returns = self.model.forward(model_inputs)
        value, index = expected_returns.max(1)
        # print(f"expected_returns:\n{expected_returns}")
        # print(f"max value:{value}")
        # print(f"model_inputs dimension: {torch.from_numpy(inputs).float().unsqueeze(0)}, action:{state.remaining[index[0]]}")
        # print("-----------")
        return state.remaining[index[0]]

    def compute_loss(self, batch, dataset, verbose=True):
        states, actions, rewards, next_states, dones = batch
        print(f"state\t t:{states[0].t}, qid:{states[0].qid}, remaining:{states[0].remaining}")
        print(f"action:{actions}")
        print(f"rewards:{rewards}")
        print(f"n_states\t t:{next_states[0].t}, qid:{next_states[0].qid}, remaining:{next_states[0].remaining}")
        print(f"dones: {dones}")
        # print(f"next_states1\t t:{next_states[1].t}, qid:{next_states[1].qid}, remaining:{next_states[1].remaining}")
        
        model_inputs = np.array([get_model_inputs(states[i], actions[i], dataset)\
            for i in range(len(states))])
        # print(f"model_inputs:{model_inputs[0][:5]}")
        model_inputs = torch.FloatTensor(model_inputs)
         
        rewards = np.array(rewards)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)

        # print("Model Inputs Size:", model_inputs.size())
        curr_Q = self.model.forward(model_inputs)


        stacked_arrays = []
        for i in range(len(next_states[0].remaining)):
            temp = get_model_inputs(next_states[0], next_states[0].remaining[i], dataset)
            stacked_arrays.append(temp)

        # print(f"stacked_arrays:{stacked_arrays}")
        if stacked_arrays:
            result_array = np.vstack(stacked_arrays)
            # model_inputs = np.array([get_model_inputs(next_states[0], next_states[0].remaining[i], dataset) \
            #     for i in range(len(next_states[0].remaining))])

            model_inputs = torch.FloatTensor(result_array)
            next_Q = self.model.forward(model_inputs)
            for i in range(result_array.shape[0]):
                print(next_states[0].remaining[i],"\t",  next_Q[i, 0].item())
            # print(f"next_Q:{next_Q}")
            max_next_Q = torch.max(next_Q, 1)[0].max().item()
            # print(f"max_next_Q:{max_next_Q}")
            expected_Q = rewards.squeeze(1) + (1 - dones) * self.gamma * max_next_Q
            
        else:
            expected_Q = rewards.squeeze(1)
        # print(f"expected_Q:{expected_Q}")
        # if verbose:
            # print(f"curr_Q:{curr_Q}")
            # print(f"expected_Q:{expected_Q}")
        loss = self.MSE_loss(curr_Q.squeeze(0), expected_Q.detach())
        print("-----------------------------------")
        return loss, curr_Q

    def update(self, batch_size, verbose=False):
        batch = self.replay_buffer.sample(batch_size)
        # print("batch", batch)
        loss, curr_Q = self.compute_loss(batch, self.dataset, verbose)
        train_loss = loss.float()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.swa:
            self.optimizer.swap_swa_sgd()
        # print(f"loss:{train_loss}")
        return train_loss, curr_Q