import os
from typing import Dict
from loguru import logger
import inspect
import datetime

from model.mdp import compute_reward

def create_directories(directories_list: Dict[str, str]) -> None:
    """
    Create directories from a list of directory paths if they do not already exist.
    Args:
        directories_list: A dictionary where keys are directory names
        and values are the corresponding directory paths.

    Returns:
        None
    """
    for directory_name, directory_path in directories_list.items():
    
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            logger.info(f"Created directory: {directory_path}")

# def save_run_info(cfg, run_time):
#     current_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

#     with open(cfg.metadata_file_path, 'w') as file:
#         file.write(f"# Run Information\n\n")
#         file.write(f"**Script was run on:** {current_date}\n\n")
#         file.write(f"**Run mode:** {cfg.run_mode}\n\n")
#         file.write(f"**Number of epochs:** {cfg.run_params.epochs}\n\n")
#         file.write(f"**Run time:** {run_time} seconds\n\n")
#         file.write(f"## Reward Function Definition\n\n")
#         file.write(f"```python\n")
#         file.write(inspect.getsource(compute_reward))
#         file.write(f"```\n")


def format_bias_output(data):

    output_lines = []
    for i, line in enumerate(data):
        output_lines.append(line)
        if (i + 1) % 2 == 0 and i + 1 < len(data):
            output_lines.append('-' * 45)

    formatted_output = '\n'.join(output_lines)
    return formatted_output




def save_run_info(cfg, train_run_time=0, eval_run_time=0):
    current_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Check if the metadata file already exists
    file_exists = os.path.isfile(cfg.metadata_file_path)

    # Read existing content
    existing_content = ""
    if file_exists:
        with open(cfg.metadata_file_path, 'r') as file:
            existing_content = file.read()

    source_code = inspect.getsource(compute_reward)
    compute_reward_source_code = source_code[:168] + str(cfg.r) + source_code[169:184] + str(cfg.b) + source_code[185:]
    # Open the file in write mode to update its content
    with open(cfg.metadata_file_path, 'w') as file:
        file.write(f"# Run Information\n\n")
        file.write(f"**Script was run on:** {current_date}\n\n")
        file.write(f"**Run mode:** {cfg.run_mode}\n\n")
        file.write(f"**Learning Rate:** {cfg.run_params.learning_rate}\n\n")
        file.write(f"**Number of epochs:** {cfg.run_params.epochs}\n\n")
        file.write(f"**Train Run time:** {train_run_time} seconds\n\n")
        file.write(f"**Eval Run time:** {eval_run_time} seconds\n\n")
        file.write(f"## Reward Function Definition\n\n")
        file.write(f"```python\n")
        file.write(compute_reward_source_code)
        file.write(f"```\n")
        file.write(existing_content)
