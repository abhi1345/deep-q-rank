import numpy as np
import pandas as pd
import random
from collections import deque
random.seed(2)

def compute_reward(t, relevance):
    if t == 0:
        return 0
    return relevance / np.log2(t + 1)

def get_features(qid, doc_id, dataset):
    qid, doc_id = int(qid), str(doc_id)
    df = dataset[dataset["doc_id"].str.contains(doc_id)][dataset["qid"] == qid]
    assert len(df) != 0, "Fix the dataset"
    df.drop(["qid", "doc_id", "rank"], axis=1, inplace=True)
    return df.values.tolist()[0]

def get_query_features(qid, doc_list, dataset):
    doc_set = set(doc_list)
    qid = int(qid)
    if len(doc_list) > 0:
        df = dataset[dataset["qid"] == qid][dataset["doc_id"].isin(doc_set)]
    else:
        df = dataset[dataset["qid"] == qid]
    assert len(df) != 0
    df.drop(["qid", "doc_id", "rank"], axis=1, inplace=True)
    return df.values

