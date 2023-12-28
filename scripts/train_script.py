import random
import torch
import argparse
from hydra import compose, initialize
from loguru import logger
from tqdm import tqdm 

from util import *
from model.dqn import DQNAgent
from model.mdp import BasicBuffer

def train_model(cfg):
    random.seed(cfg.run_params.seed)

    train_set = load_dataset(cfg.train_set_path, cfg.run_params.top_docs_count, cfg.train_set_df_path)
    val_set = load_dataset(cfg.val_set_path, cfg.run_params.top_docs_count, "" )

    train_buffer = BasicBuffer(30000)
    train_buffer.push_batch(train_set, 3)

    val_buffer = BasicBuffer(20000)
    val_buffer.push_batch(val_set, 3)

    agent = DQNAgent((48,), learning_rate=3e-4, buffer=train_buffer, dataset=train_set)

    y, z = [], []
    for i in tqdm(range(cfg.run_params.epochs)):
        y.append(agent.update(1, verbose=True))
        z.append(agent.compute_loss(val_buffer.sample(1), val_set, verbose=True))

    torch.save(agent.model.state_dict(), cfg.model_path)

    y = [float(x) for x in y]
    z = [float(x) for x in z]

    with open(cfg.train_output_file_path, 'w+') as f:
        f.write("Training Loss:\n")
        f.write(str(y))
    with open(cfg.validation_output_file_path, 'w+') as f:
        f.write("Validation Loss:\n")
        f.write(str(z))

    plot_MA_log10(y, cfg.run_params.window, cfg.train_loss_plot_path, label="train loss")
    plot_MA_log10(z, cfg.run_params.window, cfg.validation_loss_plot_path, label="validation loss")

def main():
    parser = argparse.ArgumentParser(description="Running train_script")
    parser.add_argument("--conf", type=str, help="Path to the config file")
    args = parser.parse_args()

    if args.conf:
        config_file = args.conf
        logger.info(f"Config file name: {config_file}")
    else:
        logger.info(
            "Please provide the name of the config file using the --conf argument. \nExample: --conf config.yaml"
        )

    initialize(config_path="config")
    cfg = compose(config_name=f"{config_file}")

    create_directories(
        cfg.directories
    )

    train_model(cfg.train_config)

if __name__ == "__main__":
    main()
