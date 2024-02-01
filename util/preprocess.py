import os 
import time
import pandas as pd
from pathlib import Path
from loguru import logger
from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from util.constants import *

def load_dataset(cfg, model_path: str, top_docs_count: int, run_mode: str, stage: str) -> pd.DataFrame:

    print(cfg.stage, cfg.run_mode)
    start_time = time.time()

    if cfg.stage == 'TRAIN' and cfg.run_mode == "RUN":
        input_file_path = TRAIN_SET_PATH
        df_path = TRAIN_DF_PATH
        queries_path = QUERIES_TRAIN_FILE_PATH

    elif cfg.stage == 'TRAIN' and cfg.run_mode == "DEBUG":
        input_file_path = DEBUG_TRAIN_SET_PATH
        df_path = DEBUG_TRAIN_DF_PATH
        queries_path = QUERIES_TRAIN_FILE_PATH

    elif cfg.stage == 'EVAL' and cfg.run_mode == "RUN":
        input_file_path = TEST_SET_PATH
        df_path = TEST_DF_PATH
        queries_path = QUERIES_TEST_FILE_PATH

    else:
        input_file_path = DEBUG_TEST_SET_PATH
        df_path = DEBUG_TEST_DF_PATH
        queries_path = QUERIES_TEST_FILE_PATH


    print(f"SET PATH: {input_file_path}")
    if os.path.isfile(df_path):
        df = pd.read_csv(df_path)
        if 'Unnamed: 0' in df.columns:
            df = df.drop('Unnamed: 0', axis = 1 )

        df['doc_id'] = df['doc_id'].astype(str)
        logger.info(f"Loaded from presaved dataframe in {df_path} in  {time.time() - start_time} seconds")
        print(df.info())
        print(df.head(2))
        return df

    corpus = {}
    with open(CORPUS_FILE_PATH, 'r', encoding='utf8') as f:
        for line in f:
            pid, passage = line.strip().split("\t")
            corpus[pid] = passage.strip()

    print('Loading queries...')
    queries = {}
    with open(queries_path, 'r', encoding='utf8') as f:
        for line in f:
            qid, query = line.strip().split("\t")
            queries[qid] = query.strip()

    model = SentenceTransformer(model_path, device='cuda:3')
    print(f'{model_path} model loaded.')

    dic = {"qid": [], "doc_id": [], "relevance": [], "bias": []}
    # dic = {"qid": [], "doc_id": [], "relevance": []}


    for i in range(1, 769):
        dic[i] = []

    with open(input_file_path, 'r', encoding='utf8') as f:
        qid_set = set()
        for line in tqdm(f, total=50289125):
            data = line.strip().split(" ")
            qid = data[0]

            if qid not in qid_set:
                qid_set.add(qid)
                row_counter = 1

            if row_counter > top_docs_count:
                continue
            else:
                #121352 Q0 5561504 2 1.0000000 gold_sbert 0.0
                doc_id, relevance = data[2], float(data[4])
                bias = float(data[6])

                dic["qid"].append(int(qid))
                dic["doc_id"].append(doc_id)
                dic["relevance"].append(relevance)
                dic["bias"].append(bias)

                vector = model.encode(f'{queries[qid]}[SEP]{corpus[doc_id]}')
                # print(len(vector))
                for i in range(1, 769):
                    dic[i].append(vector[i - 1])

                row_counter += 1

    df = pd.DataFrame(data=dic).sort_values(["qid", "relevance"], ascending=False)
    df.to_csv(df_path)

    logger.info(f"Loaded data from run file and saved to {df_path} in {time.time() - start_time} seconds")
    return df

def get_model_inputs(state, action, dataset) -> np.ndarray:
    """
    Get model inputs for the given state and action.
    
    Args:
    - state: State information.
    - action: Action information.
    - dataset (pd.DataFrame): Dataset for reference.
    
    Returns:
    - np.ndarray: Model inputs.
    """
    return np.array([state.t] + get_features(state.qid, action, dataset))

def get_multiple_model_inputs(state, doc_list, dataset) -> np.ndarray:
    """
    Get multiple model inputs for the given state, list of docs, and dataset.
    
    Args:
    - state: State information.
    - doc_list (List[Union[str, int]]): List of documents.
    - dataset (pd.DataFrame): Dataset for reference.
    
    Returns:
    - np.ndarray: Multiple model inputs.
    """
    return np.insert(get_query_features(state.qid, doc_list, dataset), 0, state.t, axis=1)

def get_features(qid, doc_id, dataset) -> List[float]:
    """
    Get features for the given query id and document id.
    
    Args:
    - qid (Union[str, int]): Query ID.
    - doc_id (Union[str, int]): Document ID.
    - dataset (pd.DataFrame): Dataset for reference.
    
    Returns:
    - List[float]: Features for the given query and document.
    """
    qid, doc_id = int(qid), str(doc_id)
    df = dataset[(dataset["doc_id"].str.contains(doc_id)) & (dataset["qid"] == qid)]
    assert len(df) != 0, "Fix the dataset"

    df_copy = df.copy()
    df_copy.drop(["qid", "doc_id", "relevance", "bias"], axis=1, inplace=True)
    df = df_copy
    return df.values.tolist()[0]

def get_query_features(qid, doc_list, dataset) -> np.ndarray:
    """
    Get query features for the given query ID, list of docs, and dataset.
    
    Args:
    - qid (Union[str, int]): Query ID.
    - doc_list (List[Union[str, int]]): List of documents.
    - dataset (pd.DataFrame): Dataset for reference.
    
    Returns:
    - np.ndarray: Query features.
    """
    doc_set = set(doc_list)
    qid = int(qid)
    if len(doc_list) > 0:
        df = dataset[dataset["qid"] == qid]
        df = df[df["doc_id"].isin(doc_set)]
    else:
        df = dataset[dataset["qid"] == qid]
    assert len(df) != 0
    df.drop(["qid", "doc_id", "relevance", "bias"], axis=1, inplace=True)
    return df.values
