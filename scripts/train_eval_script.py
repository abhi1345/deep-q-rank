import time
import argparse
from hydra import compose, initialize
from util import *
from scripts import *

def main():
    
    parser = argparse.ArgumentParser(description="Running train_eval_script")
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

    train_start_time = time.time()

    train_model(cfg.train_config)

    train_end_time = time.time()
    #---------------
    eval_model(cfg.eval_config)
    
    eval_end_time = time.time()

    train_run_time = train_end_time - train_start_time
    eval_run_time = eval_end_time - train_end_time
    save_run_info(cfg.train_config, train_run_time,  eval_run_time)


if __name__ == "__main__":
    main()

