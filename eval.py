
# Functions for Evaluating Agent Performance

from scipy.stats import kendalltau, spearmanr
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
random.seed(2)

def compute_reward(t, relevance):
    """
    Reward function for MDP
    """
    if t == 0:
        return 0
    return relevance / np.log2(t + 1)

def dcg_at_k(r, k, method=0):
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.

def ndcg_at_k(r, k, method=0):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max

def all_ndcg_values(r, k_list):
    ret = {}
    for k in k_list:
        ret[k] = ndcg_at_k(r, k)
    return ret

def compare_rankings(r1, r2):
    
    return kendalltau(r1, r2), spearmanr(r1, r2)

def get_rank(qid, doc_id, dataset):
    df = dataset[dataset["qid"] == qid][dataset["doc_id"] == doc_id]
    return int(list(df["rank"])[0])

def ranking_to_ranks(r, qid, dataset):
    """
    Turns list of doc id's into a list of relevance values
    """
    return [get_rank(qid, doc, dataset) for doc in r]

def evaluate_ranking(r, qid, k, dataset):
    """
    Takes in a ranked list of doc id's
    Returns ndcg @ k of ranking
    """
    print("getting relevance list")
    relevance_list = ranking_to_ranks(r, qid, dataset)
    print("computing ndcg")
    return ndcg_at_k(relevance_list, k)

def reward_from_query(agent, qid, df):
    """
    Run agent to rank a whole (single) query
    agent: DQN agent
    qid: string query id4
    """
    filtered_df = df.loc[df["qid"] == int(qid)].reset_index()
    remaining = list(filtered_df["doc_id"])
    state = State(0, qid, remaining)
    total_reward, t= 0, 0
    while not state.terminal:
        next_action = agent.get_action(state)
        t += 1
        remaining.remove(next_action)
        state = State(t, qid, remaining)
        reward = compute_reward(t, get_rank(qid, next_action, letor))
        total_reward += reward
    return total_reward

def get_agent_ranking(agent, qid, df):
    """
    Run agent to rank a whole (single) query and get list
    agent: DQN agent
    qid: string query id4
    """
    filtered_df = df.loc[df["qid"] == int(qid)].reset_index()
    remaining = list(filtered_df["doc_id"])
    random.shuffle(remaining)
    state = State(0, qid, remaining)
    ranking = []
    t = 0
    while len(remaining) > 0:
        next_action = agent.get_action(state, df)
        t += 1
        remaining.remove(next_action)
        state = State(t, qid, remaining)
        ranking.append(next_action)
    return ranking

def get_true_ranking(qid, dataset):
    """
    @qid: string query id
    @return List<doc_id strings>
    """
    df = dataset[dataset["qid"] == qid]
    df.sort_values(["rank"], inplace=True, ascending=False)
    return list(df["doc_id"])

def eval_agent_ndcg_single(agent, k, qid, dataset):
    """
    Evaluate your agent against a given LETOR dataset with 
    returns ndcg@k, averaged across all queries in dataset
    """
    print("getting agent ranking")
    agent_ranking = get_agent_ranking(agent, qid, dataset)
    print("running eval")
    cur_ndcg = evaluate_ranking(agent_ranking, qid, k, dataset)
    return cur_ndcg
    
def eval_agent_ndcg(agent, k, dataset):
    """
    Evaluate your agent against a given LETOR dataset with 
    returns mean ndcg@k across all queries in dataset
    """
    ndcg = 0
    qid_set = set(dataset["qid"])
    for i,qid in enumerate(qid_set):
        ndcg += eval_agent_ndcg_single(agent, k, qid, dataset)
        print("Finished qid {} ({}/{})".format(qid, i, len(qid_set)))
    answer = float(ndcg) / len(qid_set)
    print("Mean NDCG@{} of {} across {} samples".format(k, answer, len(qid_set)))
    return answer
    