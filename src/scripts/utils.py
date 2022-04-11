import pathlib
from pathlib import Path
import os
import datetime as dt
import time
from typing import List


def clean_folder(path: str):
    """Cleanes a folder.

    Args:
        path (str): Path to a folder.
    """
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        try:
            os.unlink(file_path)

        except Exception as e:

            print("Failed to delete %s. Reason: %s" % (file_path, e))


def get_time(file_path: str):

    st = os.stat(file_path)
    atime = dt.datetime.fromtimestamp(st.st_atime)
    now = dt.datetime.now()
    ago = now - dt.timedelta(minutes=1)

    return atime, ago


def slice_data(data: List[str]):

    n = 20

    if len(data) > n:
        data_generic = data[-n:]
    else:
        data_generic = data

    return data_generic


def txt_append(path: str, data):

    file_object = open(path, "a")
    file_object.write(str(data)[1:-1])
    file_object.write(",")
    file_object.close()


def check_data_type(data: List[dict]) -> bool:
    """Check if a key is present in json file.

    Args:
        data (List[dict]): Json data.

    Returns:
        bool: Returns true if a json contains a key.
    """
    if "Browser History" in data:
        return True
    else:
        return False
