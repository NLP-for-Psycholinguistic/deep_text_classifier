"""
creation_date: 4 august 2022

Description : 
- Define preprocessing
"""


import pandas as pd
import numpy as np
import logging
import os
from tqdm import tqdm


import tensorflow_hub as hub
import tensorflow_text




import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from transformers import CamembertTokenizer




class DataPreprocessor:
    def __init__(self, config:dict,) -> None:
        self.config = config
        self._logger = logging.getLogger(self.__class__.__name__)

    

    def prepare_data_for_keras_model(self, train : pd.DataFrame, test : pd.DataFrame, valid : pd.DataFrame):
        """_summary_

        Args:
            train (pd.DataFrame): _description_
            test (pd.DataFrame): _description_
            valid (pd.DataFrame): _description_

        Returns:
            train_data,
            train_label,
            test_data,
            test_label,
            valid_data,
            valid_label
        """
        self.max_vocab_size = self.config['model']['vocab_size']
        self.tokenizer = Tokenizer(oov_token='<OOV>')
        Docs = train["text"].tolist() + valid["text"].tolist()
        self.tokenizer.fit_on_texts(Docs)
        
        ## logging
        vocab_size = len(self.tokenizer.word_index)
        document_count = self.tokenizer.document_count
        self._logger.info(f"There are {document_count} documents in the corpus")
        self._logger.info(f"There are {vocab_size} words in the corpus, it will be set to {self.max_vocab_size} based on the config file")
        #reducing vocab size
        self.tokenizer.num_words = self.max_vocab_size
        allDocs_sequences = self.tokenizer.texts_to_sequences(Docs)
        
        self.max_length = max([len(x) for x in allDocs_sequences])
        self._logger.info(f"The max lenght is set to {self.max_length} based on the data analysis")
        word_index = self.tokenizer.word_index

        # preprocess text data with keras tokenizer and padding
        Y = []
        X = []
        Z = []

        for data in train, test, valid : 
            docs = data['text'].tolist()
            label = np.asarray(data["label"].tolist())
            group = np.asarray(data["group"].tolist())
            sequences = self.tokenizer.texts_to_sequences(docs)
            padded_sequences = pad_sequences(sequences, padding='post', truncating='post', maxlen=self.max_length)
            Y.append(label)
            X.append(padded_sequences)
            Z.append(group)

        train_label, test_label, valid_label = Y[0], Y[1], Y[2]
        train_data, test_data, valid_data =  X[0], X[1], X[2]
        train_group, test_group, valid_group = Z[0], Z[1], Z[2]

        return train_data, train_label, train_group, test_data, test_label, test_group, valid_data, valid_label, valid_group
        
    def keras_preprocessor(self, text_list:list)-> list:
        """
        Foction that reproduces text preprocessing base on a fitted tokenizer

        Args:
            text_list (list): in put text

        Returns:
            list: list of token id 
        """

        sequences = self.tokenizer.texts_to_sequences(text_list)
        text_preprocessed  = pad_sequences(sequences, padding='post', truncating='post', maxlen=self.max_length)
        return text_preprocessed

    def prepare_data_for_bert_model(self, train, test, valid):

        self.tokenizer = Tokenizer(oov_token='<OOV>')
        Docs = train["text"].tolist() + valid["text"].tolist()
        self.tokenizer.fit_on_texts(Docs)
        allDocs_sequences = self.tokenizer.texts_to_sequences(Docs)
        self.max_length = max([len(x) for x in allDocs_sequences])
        

        # preprocess text data with keras tokenizer and padding
        Y = []
        X = []

        for data in train, test, valid : 
            Y.append(np.asarray(data["label"].tolist()))
            X.append(data['text'])
        
        train_label, test_label, valid_label = Y[0], Y[1], Y[2]
        train_data, test_data, valid_data =  X[0], X[1], X[2]


        return train_data, train_label, test_data, test_label, valid_data, valid_label

    def bert_preprocessor(self, text_list):

        
        return text_list

    def prepare_data_for_camembert_model(self, train,test, valid):

        tokenizer = CamembertTokenizer.from_pretrained("camembert/camembert-base")
        
        Y = []
        X = []

        for data in train, test, valid :
            input_ids=[]
            attention_masks=[]
            for sent in data['text'] :
                bert_inp= tokenizer.encode_plus(sent,add_special_tokens = True,truncation=True, max_length=128, padding="max_length",return_attention_mask = True)
                input_ids.append(bert_inp['input_ids'])
                attention_masks.append(bert_inp['attention_mask'])
            input_ids=np.asarray(input_ids)
            attention_masks=np.array(attention_masks)
            Y.append(np.asarray(data["label"].tolist()))
            X.append([input_ids,attention_masks])

        train_label, test_label, valid_label = Y[0], Y[1], Y[2]
        train_data, test_data, valid_data =  X[0], X[1], X[2]

        self.max_length = max([len(x) for x in train_data[0]])

        return train_data, train_label, test_data, test_label, valid_data, valid_label

    def camembert_preprocessor(self, text_list : str):
        tokenizer = CamembertTokenizer.from_pretrained("camembert/camembert-base")
        input_ids = []
        attention_masks = []
        for sent in text_list : 
            bert_inp= tokenizer.encode_plus(sent,add_special_tokens = True,truncation=True, max_length=128, padding="max_length",return_attention_mask = True)
            input_ids.append(bert_inp['input_ids'])
            attention_masks.append(bert_inp['attention_mask'])
        input_ids=np.asarray(input_ids)
        attention_masks=np.array(attention_masks)

        return [input_ids,attention_masks]


