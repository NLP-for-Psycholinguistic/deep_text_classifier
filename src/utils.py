"""
creation_date: 26 july 2022

Description : 
- group all of very common used functions to deals with config,
simple operations, data loading etc...
"""

import pandas as pd
import yaml
import os
import matplotlib.pyplot as plt


def load_config(file_path: str = "config.yaml") -> dict:
    """
    load config file into dict python oject
    """
    try:
        with open(file_path, "r") as file:
            config = yaml.safe_load(file)
    except Exception as e:
        print(f"Fail to load config file because of {e}")
        config = {}
    return config


def load_stanza_data(config: dict) -> pd.DataFrame:
    """Load the data  from stanza

    Args:
        config (dict): _description_

    Returns:
        pd.DataFrame: _description_
    """
    try:
        folder = config["data"]["nlp_folder"]
        file_path = os.path.join(folder, config["data"]["stanza_file_name"])
        data = pd.read_pickle(file_path)
    except Exception as e:
        print(f"Fail to load config file because of {e}")
        data = pd.DataFrame()
    return data


def flatten(list_of_lists: list) -> list:
    """
    Flatten a list of lists to a combined list

    Args:
        list_of_lists (list): _description_

    Returns:
        list: _description_
    """

    return [item for sublist in list_of_lists for item in sublist]


def plot_graphs(ax, history, metric: str) -> None:
    """
    plot the training history

    Args:
        history (_type_): _description_
        metric (str): _description_
    """
    ax.plot(history.history[metric])
    ax.plot(history.history["val_" + metric], "")
    ax.set_xlabel("Epochs")
    ax.set_ylabel(metric)
    ax.legend([metric, "val_" + metric])


# annotation_data = pd.read_csv("/home/robin/Data/Etude_1000/annotation/20220913_fin_label.csv")


def compute_max_len(line, annotation_data):
    code = line["code"]
    token = line["token"]
    if code in annotation_data["code"].tolist():
        max_len = annotation_data[annotation_data["code"] == code]["token_len"].values[
            0
        ]

    else:
        max_len = len(token) / 5

    return max_len


def cut_col(line, col):
    return line[col][: int(line["max_len"])]
