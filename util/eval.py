from scipy.stats import kendalltau, spearmanr
import numpy as np
import pandas as pd
import random
from sklearn.utils import shuffle

from model.mdp import *


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
    """
    Gets NDCG @ [1..10] and Mean NDCG Value
    """
    ret = []
    running_sum = 0
    for i in range(1, len(r)):
        cur_ndcg = ndcg_at_k(r, i)
        if i in k_list:
            ret.append(cur_ndcg)
        running_sum += cur_ndcg
    return ret + [float(running_sum) / len(r)]

def all_ndcg_values_plus_tau(r, k_list):
    """
    Gets NDCG @ [1..10] and Mean NDCG Value
    """
    ret = []
    running_sum = 0
    for i in range(1, len(r)):
        cur_ndcg = ndcg_at_k(r, i)
        if i in k_list:
            ret.append(cur_ndcg)
        running_sum += cur_ndcg
    return ret + [float(running_sum) / len(r)] + get_tau(r)

def get_tau(r):
    return kendalltau(r, sorted(r, reverse=True))[0]

def compare_rankings(r1, r2):
    return kendalltau(r1, r2), spearmanr(r1, r2)

def get_feature(qid, doc_id, dataset, feature):
    df = dataset[(dataset["qid"] == qid) & (dataset["doc_id"] == doc_id)]
    return float(list(df[feature])[0])

def get_feature_for_docs(r, qid, dataset, feature):
    return [get_feature(qid, doc, dataset, feature) for doc in r]

def evaluate_ranking_list(r, qid, k, dataset):
    """
    Takes in a ranked list of doc id's
    Returns ndcg @k of ranking
    """
    relevance_list = get_feature_for_docs(r, qid, dataset, "relevance")
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

        reward = compute_reward(t, get_feature(qid, next_action, letor, "relevance"), get_feature(qid, next_action, letor, "bias"))
        total_reward += reward
    return total_reward

def get_agent_ranking_list(agent, qid, df):
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
    Evaluate your agent against a given dataset with 
    returns ndcg@k, averaged across all queries in dataset
    """
    agent_ranking = get_agent_ranking_list(agent, qid, dataset)
    cur_ndcg = evaluate_ranking_list(agent_ranking, qid, k, dataset)
    return cur_ndcg

def all_ndcg_single(agent, k_list, qid, dataset):
    """
    Evaluate your agent against a given LETOR dataset with 
    returns ndcg@k, averaged across all queries in dataset
    """
    agent_ranking = get_agent_ranking_list(agent, qid, dataset)
    relevance_list = get_feature_for_docs(agent_ranking, qid, dataset, "relevance")
    return all_ndcg_values(relevance_list, k_list)

def all_error_single(agent, k_list, qid, dataset):
    """
    Returns NDCG@k list plus tau for a single qid
    """
    agent_ranking = get_agent_ranking_list(agent, qid, dataset)
    relevance_list = get_feature_for_docs(agent_ranking, qid, dataset, "relevance")
    return all_ndcg_values_plus_tau(relevance_list, k_list)

def get_tau_single(agent, qid, dataset):
    agent_ranking = get_agent_ranking_list(agent, qid, dataset)
    relevance_list = get_feature_for_docs(agent_ranking, qid, dataset, "relevance")
    return get_tau(relevance_list)
    
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

def eval_agent_final(agent, k_list, dataset):
    """
    Returns a list of average NDCG@k for each k in the k_list, plus Mean NDCG
    """
    qid_set = set(dataset["qid"])
    ndcg_list = np.append(np.zeros(len(k_list)), 0)
    for qid in qid_set:
        ndcg_list += np.array(all_ndcg_single(agent, k_list, qid, dataset))
    ndcg_list /= len(qid_set)
    # print("NDCG Values: {}".format(ndcg_list))
    return ndcg_list

def get_all_errors(agent, k_list, dataset):
    """
    Returns NDCG@k List, Kendall's Tau, and Precision @ k
    """
    qid_set = set(dataset["qid"])
    ndcg_list = np.zeros(len(k_list)+2)
    for qid in qid_set:
        ndcg_list += np.array(all_error_single(agent, k_list, qid, dataset))
    ndcg_list /= len(qid_set)
    print("NDCG Values: {}".format(ndcg_list))
    return ndcg_list

def get_just_tau(agent, dataset):
    """
    Returns Kendall's Tau
    """
    qid_set = set(dataset["qid"])
    avg_tau = 0.0
    for qid in qid_set:
        avg_tau += get_tau_single(agent, qid, dataset)
    avg_tau /= len(qid_set)
    print("Tau Value: {}".format(avg_tau))
    return avg_tau
    
def write_trec_results(agent, dataset, feature_name, output_file_path: str):    
    with open(output_file_path, 'a+') as file:
        for qid in set(dataset["qid"]):
            agent_ranking = get_agent_ranking_list(agent, qid, dataset)
            for rank, doc_id in enumerate(agent_ranking, start=1):
                relevance_score = dataset[(dataset["qid"] == qid) & (dataset["doc_id"] == doc_id)][feature_name].values[0]
                file.write(f"{qid} QO {doc_id} {rank} {relevance_score} ModelName\n")

