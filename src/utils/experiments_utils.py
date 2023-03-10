# Fernando López Gavilánez, 2023

import os


def check_path(folder_name):
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)