import os
import sys
import logging
import pickle
from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        print(f"Attempting to save object at: {file_path}")
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        print(f"Object saved successfully at: {file_path}")
    except Exception as e:
        print(f"Error: {e}")
