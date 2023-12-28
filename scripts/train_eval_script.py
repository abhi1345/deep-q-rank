import argparse
from hydra import compose, initialize
from util import *
from scripts import *

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
    eval_model(cfg.eval_config)

if __name__ == "__main__":
    main()

