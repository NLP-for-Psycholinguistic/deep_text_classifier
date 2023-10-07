"""
creation_date: 4 august 2022

Description : 
- Define modelBuilder
"""


import pandas as pd
import numpy as np
import os
import logging


import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    SpatialDropout1D,
    Dropout,
    Convolution1D,
    Conv1D,
)
from tensorflow.keras.layers import LSTM, GlobalMaxPooling1D, AveragePooling1D
from tensorflow.keras.layers import Embedding
from keras.layers import LeakyReLU


from tensorflow.keras.layers import Concatenate, Input, GRU, Flatten
from tensorflow.keras.layers import Bidirectional, GlobalAveragePooling1D
from tensorflow.keras import regularizers

from src.utils_models import Attention, TransformerBlock, TokenAndPositionEmbedding

from transformers import TFCamembertModel


def focal_loss(gamma=2.0, alpha=0.25):
    def loss(y_true, y_pred):
        # Compute binary cross entropy
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred, from_logits=True)

        # Compute the modulating factor
        pt = tf.math.exp(-bce)
        modulating_factor = (1 - pt) ** gamma

        # Compute the final loss
        fl = modulating_factor * bce * alpha

        return tf.reduce_mean(fl)

    return loss


class ModelBuilder:
    def __init__(self, config: dict, label_list: list, seq_max_length: int) -> None:
        self.config = config
        self._logger = logging.getLogger(self.__class__.__name__)

        # model_parameters
        self.label_list = label_list
        self.max_length = seq_max_length
        self.num_classes = len(self.label_list)
        self.max_vocab_size = int(self.config["model"]["vocab_size"])
        self.input_dim = int(self.max_vocab_size + 1)

        # training_parameters
        self.model_kind = self.config["model"]["kind"]
        self.model_param = self.config["model"][self.model_kind]
        if self.model_kind in ["lstm", "cnn", "transformer", "attention"]:
            self.output_dim = self.model_param["embedding_size"]
        else:
            self.output_dim = None
        self.learning_rate = self.model_param["learning_rate"]

        # model
        if self.model_kind == "lstm":
            self.model = self.build_lstm_model()
        elif self.model_kind == "cnn":
            self.model = self.build_cnn_model()
        elif self.model_kind == "attention":
            self.model = self.build_attention_model()
        elif self.model_kind == "custom":
            self.model = self.build_custom_model()
        elif self.model_kind == "transformer":
            self.model = self.build_transformer_model()
        elif self.model_kind == "distilbert_multi_cased":
            self.model = self.finetune_distilbert_model()
        elif self.model_kind == "labse":
            self.model = self.finetune_laBSE_model()
        elif self.model_kind == "camembert":
            self.model = self.finetune_camembert_model()

        else:
            self._logger.warning(
                "Model kind is not supported ! Please choose between : cnn, lstm, attention, or custom"
            )

        if self.config["model"]["weight_decay"]:
            self.add_regularisation()

    def get_metrics(self) -> list:
        """Return the metrics list

        Returns:
            list: list of keras metrics
        """

        metrics = [
            tf.keras.metrics.AUC(
                name="auc", curve="PR", num_thresholds=200, from_logits=True
            ),
            tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(class_id=0, name="precision_0"),
            tf.keras.metrics.Precision(class_id=1, name="precision_1"),
            tf.keras.metrics.Recall(class_id=0, name="recall_0"),
            tf.keras.metrics.Recall(class_id=1, name="recall_1"),
        ]

        return metrics

    def add_regularisation(self):
        for layer in self.model.layers:
            if hasattr(layer, "kernel_regularizer"):
                layer._kernel_regularizer = regularizers.l2(0.02)

    def get_loss(self):
        if self.config["model"]["focal_loss"]:
            self._logger.info(" Focal Loss is selected !")
            a = self.config["model"]["alpha_focal_loss"]
            gamma = self.config["model"]["gamma_focal_loss"]
            loss = tf.keras.losses.BinaryFocalCrossentropy(
                apply_class_balancing=True, alpha=a, gamma=gamma, from_logits=False
            )
        else:
            loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        return loss

    def build_lstm_model(self):
        """

        Build a simple LSTM network

        Returns:
            model (tf.keras.model): a compile model
        """

        model = Sequential()
        model.add(
            Embedding(self.input_dim, self.output_dim, input_length=self.max_length)
        )
        model.add(SpatialDropout1D(self.model_param["spatial_dropout"]))
        model.add(LSTM(self.model_param["units_1"], return_sequences=True))
        model.add(
            LSTM(
                self.model_param["units_2"],
            )
        )
        model.add(Dropout(self.model_param["dropout"]))
        model.add(Dense(self.num_classes, activation="softmax"))

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        loss = self.get_loss()
        metrics = self.get_metrics()

        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        self._logger.info(f"Compiling the model...")
        self._logger.info(f"{model.summary()}")
        self._logger.info("Model is builded")

        return model

    def build_cnn_model(self):
        """Build very simple cnn model

        Returns:
            _type_: _description_
        """

        model = Sequential()
        model.add(
            Embedding(self.input_dim, self.output_dim, input_length=self.max_length)
        )
        model.add(SpatialDropout1D(self.model_param["spatial_dropout"]))
        model.add(
            Convolution1D(
                filters=self.model_param["nb_filters"],
                padding="same",
                kernel_size=self.model_param["kernel_size"],
                activation="relu",
            )
        )

        if self.model_param["pooling"] == "max":
            model.add(GlobalMaxPooling1D())
        elif self.model_param["pooling"] == "average":
            # model.add(GlobalAveragePooling1D())
            model.add(AveragePooling1D(padding="same", strides=1))
            model.add(Flatten())

        elif self.model_param["pooling"] == "flatten":
            model.add(Flatten())

        # flatten ?
        model.add(Dense(100, activation="relu"))

        model.add(Dropout(self.model_param["dropout"]))
        model.add(Dense(self.num_classes, activation="softmax"))

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08
        )
        loss = self.get_loss()

        metrics = self.get_metrics()

        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        self._logger.info(f"Compiling the model...")
        self._logger.info(f"{model.summary()}")
        self._logger.info("Model is builded")

        return model

    def build_attention_model(self):
        """Build very simple attention model for text classification model

        Returns:
            _type_: _description_
        """

        sequence_input = Input(shape=(self.max_length,), dtype="int32")
        embedded_sequences = Embedding(self.input_dim, self.output_dim)(sequence_input)
        lstm = Bidirectional(
            LSTM(self.model_param["lstm_units_1"], return_sequences=True),
            name="bi_lstm_0",
        )(embedded_sequences)
        (lstm, forward_h, forward_c, backward_h, backward_c) = Bidirectional(
            LSTM(
                self.model_param["lstm_units_2"],
                return_sequences=True,
                return_state=True,
            ),
            name="bi_lstm_1",
        )(lstm)
        state_h = Concatenate()([forward_h, backward_h])
        state_c = Concatenate()([forward_c, backward_c])
        context_vector, attention_weights = Attention(
            self.model_param["attention_units"]
        )(lstm, state_h)
        dense1 = Dense(20, activation="relu")(context_vector)
        dropout = Dropout(self.model_param["dropout"])(dense1)
        output = Dense(self.num_classes, activation="softmax")(dropout)

        model = tf.keras.Model(inputs=sequence_input, outputs=output)

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        metrics = self.get_metrics()

        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        self._logger.info(f"Compiling the model...")
        self._logger.info(f"{model.summary()}")
        self._logger.info("Model is builded")

        return model

    def build_transformer_model(self):
        """Build very simple transformer model for text classification
        source:(https://keras.io/examples/nlp/text_classification_with_transformer/)
        Returns:
            _type_: _description_
        """

        inputs = Input(shape=(self.max_length,))
        embedding_layer = TokenAndPositionEmbedding(
            self.max_length, self.max_vocab_size, self.output_dim
        )
        x = embedding_layer(inputs)
        transformer_block = TransformerBlock(
            self.output_dim, self.model_param["nb_heads"], self.model_param["ff_dim"]
        )
        x = transformer_block(x)
        x = GlobalAveragePooling1D()(x)
        x = Dropout(self.model_param["dropout_1"])(x)
        x = Dense(self.model_param["dense_units"], activation="relu")(x)
        x = Dropout(self.model_param["dropout_2"])(x)
        outputs = Dense(self.num_classes, activation="softmax")(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        metrics = self.get_metrics()

        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        self._logger.info(f"Compiling the model...")
        self._logger.info(f"{model.summary()}")
        self._logger.info("Model is builded")

        return model

    def finetune_distilbert_model(self):
        preprocessing_layer = hub.KerasLayer(
            "https://tfhub.dev/jeongukjae/distilbert_multi_cased_preprocess/2"
        )
        bert_layer = hub.KerasLayer(
            "https://tfhub.dev/jeongukjae/distilbert_multi_cased_L-6_H-768_A-12/1",
            trainable=True,
        )

        text_input = tf.keras.layers.Input(shape=(), dtype=tf.string)

        encoder_inputs = preprocessing_layer(text_input)

        outputs = bert_layer(encoder_inputs)
        pooled_output = outputs["pooled_output"]  # [batch_size, 768].
        sequence_output = outputs["sequence_output"]
        drop = tf.keras.layers.Dropout(self.model_param["dropout"])(pooled_output)
        output = tf.keras.layers.Dense(
            len(self.label_list), activation="softmax", name="output"
        )(drop)

        model = tf.keras.Model(inputs=text_input, outputs=output)

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        metrics = self.get_metrics()

        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        self._logger.info(f"Compiling the model...")
        self._logger.info(f"{model.summary()}")
        self._logger.info("Model is builded")

        return model

    def finetune_laBSE_model(self):
        preprocessing_layer = hub.KerasLayer(
            "https://tfhub.dev/jeongukjae/smaller_LaBSE_15lang_preprocess/1"
        )
        bert_layer = hub.KerasLayer(
            "https://tfhub.dev/jeongukjae/smaller_LaBSE_15lang/1", trainable=True
        )

        text_input = tf.keras.layers.Input(shape=(), dtype=tf.string)
        encoder_inputs = preprocessing_layer(text_input)
        outputs = bert_layer(encoder_inputs)
        pooled_output = outputs["pooled_output"]  # [batch_size, 768].
        sequence_output = outputs["sequence_output"]
        drop = tf.keras.layers.Dropout(self.model_param["dropout"])(pooled_output)
        output = tf.keras.layers.Dense(
            len(self.label_list), activation="softmax", name="output"
        )(drop)

        model = tf.keras.Model(inputs=text_input, outputs=output)

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        metrics = self.get_metrics()

        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        self._logger.info(f"Compiling the model...")
        self._logger.info(f"{model.summary()}")
        self._logger.info("Model is builded")

        return model

    def finetune_camembert_model(self):
        model_ = TFCamembertModel.from_pretrained(
            "camembert/camembert-base", from_pt=True
        )

        input_ids = tf.keras.Input(shape=(128,), dtype="int32")
        attention_masks = tf.keras.Input(shape=(128,), dtype="int32")

        output = model_(input_ids, attention_masks)
        output = output[
            0
        ]  # this is inline in config.output_hidden_states as we want only the top head

        output = output[
            :, 0, :
        ]  #  We are only interested in <cls> or classification token of the model which can be extracted
        #  using the slice operation. Now we have 2D data and build the network as one desired.
        #  While converting 3D data to 2D we may miss on valuable info.

        output = tf.keras.layers.Dense(
            self.model_param["dense_units"], activation="relu"
        )(output)
        output = tf.keras.layers.Dropout(self.model_param["dropout"])(output)
        output = tf.keras.layers.Dense(
            len(self.label_list), activation="softmax", name="output"
        )(output)

        model = tf.keras.Model(inputs=[input_ids, attention_masks], outputs=output)
        if self.model_param["trainable_layers"] > 0:
            for layer in model.layers[: self.model_param["trainable_layers"]]:
                layer.trainable = False

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        metrics = self.get_metrics()

        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        self._logger.info(f"Compiling the model...")
        self._logger.info(f"{model.summary()}")
        self._logger.info("Model is builded")

        return model

    def build_custom_model(self):
        """TO BE COMPLEMETED
        a custom model to test new things !!
        Args:
            sellf (_type_): _description_

        Returns:
            _type_: _description_
        """

        model = tf.keras.models.Sequential(
            [
                # Embedding layer to convert integer-encoded words to dense vectors
                Embedding(self.max_vocab_size, 64, input_length=self.max_length),
                SpatialDropout1D(0.3),
                # Convolutional layers to extract features from the text
                # Conv1D(filters=32, kernel_size=3, activation='relu'),
                Conv1D(64, 3, padding="valid", activation="relu"),
                GlobalAveragePooling1D(),
                # Flatten(),
                # GlobalMaxPooling1D(),
                # Flatten(),
                # Dense layers to classify the text
                # Dense(64,),
                # now add a ReLU layer explicitly:
                # LeakyReLU(alpha=0.05),
                Dropout(0.5),
                Dense(2, activation="sigmoid"),  # activation='sigmoid')
            ]
        )
        loss = self.get_loss()
        op = tf.keras.optimizers.Adam(
            learning_rate=0.00005, beta_1=0.9, beta_2=0.999, epsilon=1e-08
        )

        model.compile(optimizer=op, loss=loss, metrics=self.get_metrics())
        self._logger.info(f"Compiling the model...")
        self._logger.info(f"{model.summary()}")
        self._logger.info("Model is builded")

        return model
