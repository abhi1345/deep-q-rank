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
from scipy.stats import kendalltau
import os
import argparse

# Internal Packages
from preprocess import *
from dqn import *
from mdp import *
from eval import *

# BEGIN USER VARIABLES (CHANGE THESE)

LETOR_PATH = "/home/u27948/data"
OUTPUT_FILE_NAME = time.asctime() + " eval output.txt"
PRETRAINED_MODEL_PATH = "/home/u27948/deep-q-rank/best_model.pth"
NDCG_K_LIST = [1,2,3,4,5,6,7,8,9,10]
EPOCHS = 1000

# END USER VARIABLES

def train_model(fold):
    # Load in Data
    train_set = load_letor(LETOR_PATH + "/Fold{}/train.txt".format(fold))
    train_buffer = BasicBuffer(100000)
    while len(train_buffer) < 100000:
        train_buffer.push_batch(train_set, 1)

    # Instantiate agent
    agent = DQNAgent((47,), 
                     learning_rate=3e-4, 
                     buffer=train_buffer, 
                     dataset=train_set)

    # Begin Training
    for i in range(EPOCHS):
        print("Beginning Iteration {}\n".format(i))
        agent.update(1, verbose=False)

    # Save Model
    model_name = time.asctime() + " fold{} trained model.pth".format(fold)
    torch.save(agent.model.state_dict(), model_name)
    print("Saved Model for fold{}".format(fold))
    
    # Get Training Errors
    error_list = get_all_errors(agent, NDCG_K_LIST, train_set)
    print("Fold {} train error's: {}".format(fold, error_list))
    with open(OUTPUT_FILE_NAME, "a+") as f:
        f.write("Train Error List Fold {}\n".format(fold))
        f.write("{}\n".format(error_list))
    
    return agent
    
        
def val_model(agent, fold):
    # Load in Data
    val_set = load_letor(LETOR_PATH + "/Fold{}/vali.txt".format(fold))
    val_buffer = BasicBuffer(100000)
    while len(val_buffer) < 100000:
        val_buffer.push_batch(val_set, 1)
        
    error_list = get_all_errors(agent, NDCG_K_LIST, val_set)
    print("Fold {} val error's: {}".format(fold, error_list))
    with open(OUTPUT_FILE_NAME, "a+") as f:
        f.write("Val Error List Fold {}\n".format(fold))
        f.write("{}\n".format(error_list))
        
    

def test_model(agent, fold):
    # Load in Data
    test_set = load_letor(LETOR_PATH + "/Fold{}/test.txt".format(fold))
    test_buffer = BasicBuffer(100000)
    while len(test_buffer) < 100000:
        test_buffer.push_batch(test_set, 1)
        
    error_list = get_all_errors(agent, NDCG_K_LIST, test_set)
    print("Fold {} val error's: {}".format(fold, error_list))
    with open(OUTPUT_FILE_NAME, "a+") as f:
        f.write("Test Error List Fold {}\n".format(fold))
        f.write("{}\n".format(error_list))
            
    print("Finished Successfully Evaluating Model.")
    print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process fold value.')
    parser.add_argument('--fold', help='fold help')
    args = parser.parse_args()
    FOLD = args.fold
    agent = train_model(FOLD)
    val_model(agent, FOLD)
    test_model(agent, FOLD)

