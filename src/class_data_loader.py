"""
creation_date: 26 july 2022


Description : 
- Define DataLoader
"""

import pandas as pd
import numpy as np
import os
import logging
import pprint
import random

from collections import Counter
from src.utils import flatten, cut_col

from sklearn.model_selection import StratifiedShuffleSplit


class DataLoader:
    def __init__(self, config, filter=False) -> None:
        self.config = config
        self._logger = logging.getLogger(self.__class__.__name__)

        self.data_folder = self.config["data"]["nlp_folder"]
        self.file_path = os.path.join(self.data_folder, config["data"]["nlp_file_name"])

        self.label = self.config["data"]["target"]
        self.text_col = self.config["data"]["text_col"]
        self.id_col = self.config["data"]["id_col"]

        # filter
        if filter:
            self.subgroup = self.config["data"]["filter"]["subgroup"]

            self.pos_filter = self.config["data"]["filter"]["pos_filter"]

            self.max_len = self.config["data"]["filter"]["max_number_word"]
            self.filter_hapax = self.config["data"]["filter"]["filter_hapax"]

        # sequence
        self.seq_len = int(self.config["data"]["sequence"]["size"])
        self.overlap = self.config["data"]["sequence"]["overlap"]
        self.len_train_indices = int(15000 / self.seq_len)
        # print(self.len_train_indices)

        # split
        self.resample = self.config["data"]["split"]["upsampling"]
        self.test_size = self.config["data"]["filter"]["test_size"]
        self.seed = self.config["data"]["filter"]["seed"]

        self.data = self.load()
        self.hapax = self.compute_hapax()
        if filter:
            self.data = self.filter()

        self.min_prop = int(len(self.data) / len(self.data[self.data[self.label] == 0]))
        print(self.min_prop)

    def load(self) -> pd.DataFrame:
        """Load the data file

        Returns:
            pd.DataFrame: _description_
        """
        try:
            # load textual data
            data = pd.read_pickle(self.file_path)[
                [self.id_col, self.text_col, self.label]
            ]

        except Exception as e:
            self._logger.error(f"Fail to load data file because of {e}")
            data = pd.DataFrame()
        return data

    def filter_token(self, x: list) -> list:
        """
        filter the pos tag based on self.pos_filter

        Args:
            x (list): _description_
        Returns:
            filter_token (list) : filtered token list
        """
        filter_token = []
        for token in x:
            if (
                token[1] in self.pos_filter and len(self.pos_filter) > 0
            ):  # is in selected pos tag
                if self.filter_hapax:
                    if token[0] in set(self.hapax):
                        filter_token.append("<OOV>")

                    else:
                        filter_token.append(token[0])
                    continue
                else:
                    filter_token.append(token[0])

            else:
                filter_token.append(token[1])
        self._logger.info(f"The tokens were filter based on pos tag: {self.pos_filter}")
        return filter_token

    def filter(self) -> pd.DataFrame:
        """
        filter the data

        Returns:
            pd.DataFrame: _description_
        """
        try:
            filter_data = self.data[
                self.data[self.subgroup[0]].isin(self.subgroup[1])
            ].reset_index()
            if len(self.pos_filter) > 0:
                filter_data[self.text_col] = filter_data["pos"].apply(
                    lambda x: self.filter_token(x)
                )
            for col in ["token", "lemma", "pos", "morph"]:
                filter_data[col] = filter_data.apply(lambda x: cut_col(x, col), axis=1)

        except Exception as e:
            self._logger.error(f"Fail to filter file file because of {e}")
            filter_data = pd.DataFrame()
        return filter_data

    def get_indices_list(self, word_list: list) -> list:
        k = int(len(word_list) / self.seq_len)
        r = len(word_list) % self.seq_len
        indices_list = []
        for i in range(k):
            s = i * self.seq_len
            if s > self.overlap:
                s = s - self.overlap
            e = (i + 1) * self.seq_len
            indices_list.append((s, e))
        e = indices_list[-1][1]
        indices_list.append((e, r))
        return indices_list

    def get_train_indices_list(self, word_list, label=1, upsampling_0=False):
        np.random.seed(self.seed)
        max_len = len(word_list)
        if label == 0:
            if upsampling_0:
                n_indices = self.len_train_indices * self.min_prop  # we take
            else:
                n_indices = self.len_train_indices
        else:
            n_indices = self.len_train_indices
        # print("max_len", max_len)
        # print("seq_len", self.seq_len)
        # print("n_indices", n_indices)
        I = np.sort(np.random.randint(0, max_len - self.seq_len, n_indices))
        indices_list = [(i, i + self.seq_len) for i in I]
        return indices_list

    def get_sequence(
        self, filter_data, training=False, upsampling_0=False
    ) -> pd.DataFrame:
        sequenced_data = pd.DataFrame()
        GROUP = []
        LABEL = []
        TEXT = []
        for i in range(len(filter_data)):
            line = filter_data.loc[i]
            word_list = line[self.text_col]
            label = line[self.label]
            code = line[self.id_col]
            if training:
                indices_list = self.get_train_indices_list(
                    word_list=word_list, label=label, upsampling_0=upsampling_0
                )
                # print(code, indices_list)
            else:
                indices_list = self.get_indices_list(word_list=word_list)
            for indices in indices_list:
                start = indices[0]
                end = indices[1]
                text = " ".join(word_list[start:end]).replace("  ", " ").strip().lower()
                TEXT.append(text)
                LABEL.append(label)
                GROUP.append(code)
        # store in dataframe
        sequenced_data["text"], sequenced_data["label"], sequenced_data["group"] = (
            TEXT,
            LABEL,
            GROUP,
        )
        # sequenced_data['text'], sequenced_data['label'] = TEXT, LABEL

        return sequenced_data.sample(frac=1).reset_index(drop=True)

    def compute_hapax(self) -> list:
        """
        compute the hapax list od the dataset
        """

        hapax = [
            elt
            for elt in Counter(flatten(self.data[self.text_col].tolist())).values()
            if elt == 1
        ]

        return hapax

    def save_data_description(self, saving_folder: str) -> str:
        """
        save data descritption related to current training

        Args:
            saving_folder (str): folder to save the description

        Returns:
            str: _description_
        """
        text = ""
        saving_filepath = os.path.join(saving_folder, "data_description.txt")
        for elt in {"ensemble": self.data, "filter": self.data}.items():
            text = text + elt[0] + "\n"
            text = text + "## Data description" + "\n"
            d = elt[1]
            hapax = len(
                [
                    elt
                    for elt in Counter(flatten(d[self.text_col].tolist())).values()
                    if elt == 1
                ]
            )
            text = (
                text
                + f"Il y a {len(d)} documents, de longeur moyenne {d[self.text_col].apply(len).mean()}"
                + "\n"
            )
            text = text + f"Il y a  {d[self.text_col].apply(len).sum()} token" + "\n"
            text = (
                text
                + f"Il y a {len(set(flatten(d[self.text_col].tolist())))} mots uniques"
                + "\n"
            )
            text = text + f"Il y a {hapax}  hapax (qui n'apparaisse qu'une fois)" + "\n"
            text = text + "LABELS DESCRIPTION: " + "\n" + f"{self.label}"
            text = text + pprint.pformat(d[self.label].value_counts()) + "\n"

        with open(saving_filepath, "w") as f:
            f.write(text)

        self._logger.info(f"Data description save in {saving_filepath}")

    def get_train_test_valid(self) -> tuple:
        """
        split data in train and test set

        Returns:
            tuple: train, test, valid
        """

        data = self.data
        # test on random
        if self.config["data"]["split"]["random_label"]:
            data[self.label] = np.random.choice(
                [0, 1], size=(len(data),), p=[0.25, 0.75]
            )
            data["strati_target"] = data[self.label].astype(
                str
            )  # + "_" + data["sexe"].astype(str) + "_" + data["cat_tem"].astype(str)
        else:
            data["strati_target"] = data[self.label].astype(
                str
            )  # + "_" + data["cat_tem"].astype(str) +"_" + data["sexe"].astype(str)
        gss = StratifiedShuffleSplit(
            n_splits=1, test_size=self.test_size, random_state=self.seed
        )
        for remaining_index, test_index in gss.split(
            data[self.text_col], data["strati_target"]
        ):
            remaining = data.loc[remaining_index].reset_index(drop=True)
            test = data.loc[test_index].reset_index()

        gss_valid = StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=self.seed
        )

        for train_index, valid_index in gss_valid.split(
            remaining[self.text_col], remaining["strati_target"]
        ):
            train = remaining.loc[train_index].reset_index(drop=True)
            valid = remaining.loc[valid_index].reset_index(drop=True)

        train = self.get_sequence(train, training=True, upsampling_0=True)
        test = self.get_sequence(test, training=True)
        valid = self.get_sequence(valid, training=True)

        if self.resample:
            self._logger.info(f"Resampling training ! ")
            from sklearn.utils import resample

            majo = train[train.label == 1]
            mino = train[train.label == 0]
            min_resample = resample(
                mino,
                replace=True,  # sample with replacement
                n_samples=majo.shape[0],  # to match majority class
                random_state=self.seed,
            )
            train = pd.concat([majo, min_resample]).sample(frac=1)

        self._logger.info(f'Train label:  {train["label"].value_counts()}')
        self._logger.info(
            f"\n There are {len(train)}, training examples, \n There are {len(valid)}, validation examples, \n there are {len(test)} testing example"
        )

        return train, test, valid

    def save_data_to_hyperdeep(self, train, test, valid):
        return NotImplementedError
