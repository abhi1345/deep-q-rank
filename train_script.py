# BEGIN USER VARIABLES (CHANGE THESE)

path_to_training_set = "/Users/Abhi/Desktop/development/MQ2008-list/Fold1/train.txt"
is_above_path_a_directory = False # Set to true if training on multiple sets
path_to_val_set = "/Users/Abhi/Desktop/development/MQ2008-list/Fold1/vali.txt"
num_iterations = 1
output_file_name = "output.txt"

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
from util import *
from eval import *

def main():
    # Load in Data 
    train_set = load_letor(path_to_training_set)
    val_set = load_letor(path_to_val_set)

    trainBuffer = BasicBuffer(30000)
    trainBuffer.push_batch(train_set, 3)

    validationBuffer = BasicBuffer(20000)
    validationBuffer.push_batch(val_set, 3)

    # Instantiate agent
    agent = DQNAgent((47,), learning_rate=3e-4, buffer=trainBuffer, dataset=train_set)

    # Begin Training
    y, z = [], []
    for i in range(num_iterations):
        y.append(agent.update(1, verbose=True))
        z.append(agent.compute_loss(validationBuffer.sample(1), val_set, verbose=True))

    # Save Model
    model_name = time.asctime() + " model.pth"
    torch.save(agent.model.state_dict(), model_name)

    # Write Losses to File
    with open(output_file_name, 'w+') as f:
        f.write("Training Loss:\n")
        f.write(str([float(x) for x in y]))
        f.write("\n\n")
        f.write("Validation Loss:\n")
        f.write(str([float(x) for x in z]))
        f.write("\n")

if __name__ == "__main__":
    main()
