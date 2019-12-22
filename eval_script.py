# External Packages
import numpy as np
import pandas as pd
import random
from sklearn.utils import shuffle
import torch
import torch.nn as nn
import torch.autograd as autograd
from torchcontrib.optim import SWA
from collections import deque
import matplotlib.pyplot as plt
import time
random.seed(2)

# Internal Packages
from preprocess import *
from dqn import *
from mdp import *
from eval import *

# BEGIN USER VARIABLES (CHANGE THESE)

LETOR_PATH = "/Users/Abhi/Desktop/development/MQ2008-list/"
OUTPUT_FILE_NAME = time.asctime() + " eval output.txt"
PRETRAINED_MODEL_PATH = "Sat Dec 21 20:54:35 2019 model.pth"

# END USER VARIABLES

def train_model(LETOR_PATH, OUTPUT_FILE_NAME, PRETRAINED_MODEL_PATH):
    print("Creating Agent from Saved Model")
    model = DQN((47,), 1)
    model.load_state_dict(torch.load(PRETRAINED_MODEL_PATH))
    agent = DQNAgent((47,), learning_rate=3e-4, buffer=None, dataset=None, pre_trained_model=model)

    for fold in [1,2,3,4,5]:
        print("\nRunning Fold {}".format(fold))
        print("Loading LETOR TEST SET")
        test_path = LETOR_PATH + "/Fold{}".format(fold) + "/test.txt"
        letor_test = load_letor(test_path)
        print("Running Eval on LETOR Test")
        ndcg_list = all_ndcg_values(agent, 1, letor_test)
        print("Saving results")
        with open(OUTPUT_FILE_NAME, "a+") as f:
            f.write(str(ndcg))
            f.write("\n")

if __name__ == "__main__":
    train_model(LETOR_PATH, OUTPUT_FILE_NAME, PRETRAINED_MODEL_PATH)
