import torch
import argparse
import time
from hydra import compose, initialize
from loguru import logger 
from tqdm import tqdm
import os
import ir_measures, ir_datasets
from ir_measures import *
import subprocess

from util import *
from util.constants import *
from model.dqn import DQN, DQNAgent

def eval_model(cfg):

    trec_output_file_path = cfg.eval_trec_output_file_path
    ndgc_output_file_path = cfg.eval_ndgc_output_file_path

    fold_list = cfg.fold_list
    ndcg_k_list = cfg.ndcg_k_list

    logger.info("Loading TEST SET")
    test_set = load_dataset(cfg, cfg.bert_model_path, cfg.run_params.top_docs_count, cfg.run_mode, "EVAL")
    if 'Unnamed: 0' in test_set.columns:
        test_set = test_set.drop('Unnamed: 0', axis = 1)
    print(f"test_set columns: {test_set.columns}")


    start_time = time.time()
    logger.info("Creating Agent from Saved Model")
    model = DQN((769,), 1)
    model.load_state_dict(torch.load(cfg.pretrained_model_path))

    learning_rate = cfg.run_params.learning_rate
    agent = DQNAgent((769,), learning_rate=learning_rate, buffer=None, dataset=None, pre_trained_model=model)
    MRR10_input = calculate_MRR(qrel_file, test_set, 10)

    for fold in fold_list:

        logger.info("Running Eval on test dataset with Fold {}".format(fold))
        ndcg_list = eval_agent_final(agent, ndcg_k_list, test_set)
        print(test_set.columns)
        write_trec_results(agent, test_set, ["relevance", "bias"], trec_output_file_path )

        logger.info(f"Saving TREC results to {trec_output_file_path}")
        #MRR
        # qrel_file = "/home/shiva_soleimany/RL/deep-q-rerank/data/qrels.dev.small.tsv"
        

        MRR10_output = calculate_MRR(qrel_file, trec_output_file_path, 10)

        bias_values = calculate_bias(trec_output_file_path)
        formatted_bias_values = format_bias_output(bias_values) 

        with open(ndgc_output_file_path, "w") as f:
            f.write("Fold {} NDCG Values: {}\n".format(fold, ndcg_k_list))
            f.write(str(ndcg_list))
            f.write("\n")
            f.write(f"Fold {fold} input MRR@10 Value: {MRR10_input}\n")
            f.write(f"Fold {fold} output MRR@10 Value: {MRR10_output}\n")
            f.write(f"Fold {fold} output bias Value:\n{formatted_bias_values}\n")
        logger.info(f"Saving NDCG results to {ndgc_output_file_path}")

    logger.info("Finished Evaluating Model Successfully.")
    logger.info("--- %s seconds ---" % (time.time() - start_time))

def main():

    parser = argparse.ArgumentParser(description="Running eval_script")
    parser.add_argument("--conf", type=str, help="Path to the config file")
    args = parser.parse_args()

    if args.conf:
        config_file = args.conf
        logger.info(f"Config file name: {config_file}")
    else:
        logger.info(
            "Please provide the name of the config file using the --conf argument. \nExample: --conf rank.yaml"
        )

    initialize(config_path="config")
    cfg = compose(config_name=f"{config_file}")

    create_directories(
        cfg.directories
    )

    start_time = time.time()
    eval_model(cfg.eval_config)
    end_time = time.time()

    eval_run_time = end_time - start_time
    save_run_info(cfg.train_config, 0, eval_run_time)

if __name__ == "__main__":
    main()
