# State and Buffer Classes

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

def compute_reward(t, r, relevance, b, bias):
    """
    Bias:[-1, 1]-> abs or lambda*bias. Relevance:[0, 1]
    """    
    if t == 0:
        return 0
    reward = (r * relevance - b * abs(bias)) / np.log2(t + 1)
    # print(f"t:{t},r:{r}, rel:{relevance}, b:{b}, bias:{bias}, reward:{reward}")
    return reward 

class State:

    def __init__(self, t, query, remaining):
        # print("state init")
        self.t = t
        self.qid = query #useful for sorting buffer
        self.remaining = remaining

    def pop(self):
        print("pop")
        return self.remaining.pop()

    def initial(self):
        return self.t == 0

    def terminal(self):
        print("terminal")
        return len(self.remaining) == 0

class BasicBuffer:

    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, np.array([reward]), next_state, done)
        self.buffer.append(experience)

    def push_batch(self, df, relevance_coef, bias_coef, n):
        # print(f"push_batch-> relevance_coef:{relevance_coef}, bias_coef:{bias_coef}")
        for i in range(n):
            random_qid = random.choice(list(df["qid"]))
            filtered_df = df.loc[df["qid"] == int(random_qid)].reset_index()
            row_order = [x for x in range(len(filtered_df))]
            X = [x[1]["doc_id"] for x in filtered_df.iterrows()]
            # print(f"X before:{X}")
            random.shuffle(row_order)
            for t,r in enumerate(row_order):
                cur_row = filtered_df.iloc[r]
                old_state = State(t, cur_row["qid"], X[:])
                
                action = cur_row["doc_id"]
                # print(f"action:{action}")
                X.remove(action)
                # print(f"X after:{X}")
                new_state = State(t+1, cur_row["qid"], X[:])
# 
                reward = compute_reward(t+1, relevance_coef, cur_row["relevance"], bias_coef, cur_row["bias"])
                self.push(old_state, action, reward, new_state, t+1 == len(row_order))
                filtered_df.drop(filtered_df.index[[r]])

    def sample(self, batch_size):
        # print("sample")
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        return (state_batch, action_batch, reward_batch, 
            next_state_batch, done_batch)

    def __len__(self):
        return len(self.buffer)
