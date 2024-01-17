import torch
import argparse
import time
from hydra import compose, initialize
from loguru import logger 

from util import *
from model.dqn import DQN, DQNAgent

def eval_model(cfg):

    test_set_path = cfg.test_set_path
    trec_output_file_path = cfg.eval_trec_output_file_path
    ndgc_output_file_path = cfg.eval_ndgc_output_file_path
    pretrained_model_path = cfg.pretrained_model_path
    fold_list = cfg.fold_list
    ndcg_k_list = cfg.ndcg_k_list

    logger.info("Loading TEST SET")
    test_set = load_dataset(test_set_path, cfg.run_params.top_docs_count, '')

    start_time = time.time()
    logger.info("Creating Agent from Saved Model")
    model = DQN((48,), 1)
    model.load_state_dict(torch.load(pretrained_model_path))
    agent = DQNAgent((48,), learning_rate=3e-4, buffer=None, dataset=None, pre_trained_model=model)

    for fold in fold_list:

        logger.info("Running Eval on test dataset with Fold {}".format(fold))
        ndcg_list = eval_agent_final(agent, ndcg_k_list, test_set)
        write_trec_results(agent, test_set, "relevance", trec_output_file_path )
        logger.info("Saving results")
        with open(ndgc_output_file_path, "w") as f:
            f.write("Fold {} NDCG Values: {}\n".format(fold, ndcg_k_list))
            f.write(str(ndcg_list))
            f.write("\n")
            
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

    eval_model(cfg.eval_config)

if __name__ == "__main__":
    main()
