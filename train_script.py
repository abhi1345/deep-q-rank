# BEGIN USER VARIABLES (CHANGE THESE)

TRAINING_SET_PATHS = "/home/u27948/data/Fold1/train.txt"
IS_TRAIN_SET_DIR = False # Set to true if training on multiple sets in a directory
VAL_SET_PATH = "/home/u27948/data/Fold1/vali.txt"
EPOCHS = 10000
OUTPUT_FILE_NAME = "/home/u27948/output/losses.txt"

# END USER VARIABLES

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

def main():
    # Load in Data
    train_set = load_letor(TRAINING_SET_PATH)
    val_set = load_letor(VAL_SET_PATH)

    train_buffer = BasicBuffer(30000)
    train_buffer.push_batch(train_set, 3)

    val_buffer = BasicBuffer(20000)
    val_buffer.push_batch(val_set, 3)

    # Instantiate agent
    agent = DQNAgent((47,), learning_rate=3e-4, buffer=train_buffer, dataset=train_set)

    # Begin Training
    y, z = [], []
    for i in range(EPOCHS):
        print("Beginning Iteration {}\n".format(i))
        y.append(agent.update(1, verbose=True))
        z.append(agent.compute_loss(val_buffer.sample(1), val_set, verbose=True))

    # Save Model
    model_name = time.asctime() + " model.pth"
    torch.save(agent.model.state_dict(), model_name)

    # Write Losses to File
    with open(OUTPUT_FILE_NAME, 'w+') as f:
        f.write("Training Loss:\n")
        f.write(str([float(x) for x in y]))
        f.write("\n\n")
        f.write("Validation Loss:\n")
        f.write(str([float(x) for x in z]))
        f.write("\n")

if __name__ == "__main__":
    main()
