import os
from typing import Dict
from loguru import logger

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