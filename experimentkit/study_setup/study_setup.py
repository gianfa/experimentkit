"""
Setup a project according to a typical ML experiment structure
"""
import os
from typing import Optional, Union

import experimentkit.study_setup.project_structures as proj_struct
from experimentkit._config import *


# TODO: convert os to pathlib functions
def create_folders_from_structure_dict(
        dir_structure: dict,
        parent_directory: Union[Path, str]) -> Path:
    """
    Create a hierarchical folder structure based on the 
    `dir_structure` dictionary in a parent directory.

    Examples
    --------
    >>> import experimentkit.study_setup.project_structures as proj_struct
    >>> parent_folder = "my/parent/folder"
    >>> create_folders_from_structure_dict(proj_struct.DATA_SCIENCE, parent_folder)
    # "my/parent/folder"
    >>> dir_structure = {
    ...     'reports': {
    ...         'imgs': None,
    ...         '__README.md__': None
    ...     }
    ... }
    >>> parent_folder = "my/parent/folder"
    >>> create_folders_from_structure_dict(dir_structure, parent_folder)
    # "my/parent/folder"
    """
    for key, value in dir_structure.items():
        # If is None: create something
        if value is None:
            # If it's a special string, e.g. __STRING__, call the function
            if key.startswith("__") and key.endswith("__"):
                fname = key[2:-2]
                fpath = os.path.join(parent_directory, fname)
                with open(fpath, "w") as file:
                    pass
            else:
                # Create a folder
                folder_path = os.path.join(parent_directory, key)
                os.makedirs(folder_path, exist_ok=True)
        # If is a dict: recursively create the subfolders
        elif isinstance(value, dict):
            sub_dir = os.path.join(parent_directory, key)
            os.makedirs(sub_dir, exist_ok=True)
            create_folders_from_structure_dict(value, sub_dir)
    return Path(parent_directory)


def init_project(
        project_name: str, 
        parent_folder: Union[Path, str],
        structure: str = "data_science") -> Optional[Path]:
    parent_folder = Path(parent_folder)
    assert parent_folder.exists(), \
        f"Parent folder '{parent_folder}' must exist, instead it doesn't"

    project_dir = parent_folder/project_name

    if type(structure) == str:
        if structure.lower() == "data_science":
            create_folders_from_structure_dict(
                proj_struct.DATA_SCIENCE, project_dir)

    return project_dir


