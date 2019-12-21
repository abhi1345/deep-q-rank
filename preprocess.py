import numpy as np
import pandas as pd
import random
from sklearn.utils import shuffle
import torch
import torch.nn as nn
import torch.autograd as autograd
from torchcontrib.optim import SWA
from collections import deque

def load_letor(filepath):
    dic = {}
    for i in range(1, 47):
    dic[i] = []
    dic["qid"] = []
    dic["rank"] = []
    dic["doc_id"] = []
    with open(filepath) as fp:
    line = fp.readline()
    cnt = 0
    while line:
        cnt += 1
        if cnt % 60000 == 0:
        print("Done with {} lines".format(cnt))
        ls, doc_id = line.strip().split("#")
        doc_id = doc_id.split("=")[1][:-5]
        dic["doc_id"].append(doc_id)
        for i,pair in enumerate(ls.split(" ")):
        if i == 0:
            dic["rank"].append(int(pair))
        elif i == 1:
            qid = int(pair.split(":")[1])
            dic["qid"].append(qid)
        elif ":" in pair:
            feature, value = pair.split(":") 
            dic[int(feature)].append(float(value))
        line = fp.readline()
    df = pd.DataFrame(data=dic).sort_values(["qid", "rank"], ascending=False)
    return df

def get_model_inputs(state, action, dataset):
    return np.array([state.t] + get_features(state.qid, action, dataset))

def get_multiple_model_inputs(state, doc_list, dataset):
    return np.insert(get_query_features(state.qid, doc_list, dataset), 0, state.t, axis=1)
