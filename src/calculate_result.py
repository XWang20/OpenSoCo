import os
import json
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--task_name", type=str)
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--result_path", type=str)

    return parser.parse_args()

args = get_args()

for root, dirs, files in os.walk(".", topdown=False):
    for name in files:
        if "test_result" in name:
            path = os.path.join(root, name)
            result = json.load(open(path, "r"))
            val_result = result["validation_result"]
