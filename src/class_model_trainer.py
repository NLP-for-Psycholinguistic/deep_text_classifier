"""
creation_date: 4 august 2022

Description : 
- Define ModelTrainer
"""


import logging
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from src.utils import plot_graphs

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    roc_auc_score,
    precision_recall_curve,
    auc,
)
from sklearn.metrics import RocCurveDisplay

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
import transformers
from src.utils_models import Attention, TransformerBlock, TokenAndPositionEmbedding

import tensorflow as tf


class ModelTrainer:
    def __init__(self, config: dict, save_path: str) -> None:
        self.config = config
        self._logger = logging.getLogger(self.__class__.__name__)
        self.n_epochs = int(self.config["model"]["training"]["epochs"])
        self.save_path = save_path
        self.batch_size = self.config["model"]["training"]["batch_size"]
        self.seed = self.config["data"]["filter"]["seed"]
        self.pos_label = self.config["model"]["pos_label"]

    def train(self, model, train_data, train_label, valid_data, valid_label):
        weights = compute_class_weight(
            class_weight=self.config["model"]["training"]["class_weight"],
            classes=np.unique(train_label),
            y=train_label,
        )
        class_weights = {i: round(weights[i]) for i in range(len(weights))}

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=10, verbose=0, mode="min"),
            ModelCheckpoint(
                os.path.join(self.save_path, "model.hdf5"),
                save_best_only=True,
                monitor="val_loss",
                mode="min",
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=6,
                verbose=1,
                epsilon=1e-4,
                mode="min",
            ),
        ]

        self.history = model.fit(
            train_data,
            to_categorical(train_label),
            epochs=self.n_epochs,
            validation_data=(valid_data, to_categorical(valid_label)),
            class_weight=class_weights,
            callbacks=callbacks,
            batch_size=self.batch_size,
        )

        self._logger.info(
            f"Model is trained ans best model is saved in {self.save_path}"
        )

        model_file_path = os.path.join(self.save_path, "model.hdf5")
        customs_layers = {
            "TFCamembertModel": transformers.TFCamembertModel,
            "TokenAndPositionEmbedding": TokenAndPositionEmbedding,
            "TransformerBlock": TransformerBlock,
            "Attention": Attention,
        }
        fitted_model = tf.keras.models.load_model(
            model_file_path, custom_objects=customs_layers
        )

        return fitted_model

    def plot_history(self, show=False):
        """
        Plot the trainign history
        """
        fig = plt.figure(figsize=(16, 6))
        ax1 = plt.subplot(1, 3, 1)
        plot_graphs(ax1, self.history, "precision_0")
        ax2 = plt.subplot(1, 3, 2)
        plot_graphs(ax2, self.history, "precision_1")
        ax3 = plt.subplot(1, 3, 3)
        plot_graphs(ax3, self.history, "loss")
        if show:
            plt.show()

        return fig

    def plot_roc_auc_curves(self, pred_keras, test_label, save=True, name=""):
        # pred_keras = model.predict(test_data)[:,1]
        # print(model.predict(test_data))
        fpr_keras, tpr_keras, thresholds_keras = roc_curve(
            test_label, pred_keras, pos_label=self.pos_label
        )
        auc_keras = auc(fpr_keras, tpr_keras)
        plt.figure(1)
        plt.plot([0, 1], [0, 1], "k--")
        plt.plot(
            fpr_keras,
            tpr_keras,
            label=str(self.seed) + " (area = {:.3f})".format(auc_keras),
        )

        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.title("ROC curve")
        plt.legend(loc="best")

        if save:
            fig_save_name = os.path.join(self.save_path, f"roc_auc_{name}.png")
            plt.savefig(fig_save_name)
        else:
            plt.show()

    def plot_micro_roc_auc_curves(self, pred_keras, test_label, save=True, name=""):
        n_classes = 2
        roc_auc_micro = 0.0
        # pred_keras = model.predict(test_data)[:,1]

        y_true_concat = np.concatenate((test_label, 1 - test_label))
        y_scores_concat = np.concatenate((pred_keras, 1 - pred_keras))
        fpr_micro, tpr_micro, _ = roc_curve(y_true_concat, y_scores_concat)
        roc_auc_micro = auc(fpr_micro, tpr_micro)
        plt.figure(2)
        #    Plot average micro ROC curve
        plt.plot(
            fpr_micro,
            tpr_micro,
            lw=2,
            label="Micro-average ROC curve (AUC = %0.2f)" % roc_auc_micro,
        )
        plt.plot([0, 1], [0, 1], "k--")

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Micro-average receiver operating characteristic example")
        plt.legend(loc="best")
        if save:
            fig_save_name = os.path.join(self.save_path, f"micro_roc_auc_{name}.png")
            plt.savefig(fig_save_name)
        else:
            plt.show()

    def plot_pr_curves(self, pred_keras, test_label, save=True, name=""):
        # Predict probabilities for the test set
        # pred_keras = model.predict(test_data)[:,1]
        # Compute precision, recall and thresholds
        precision, recall, thresholds = precision_recall_curve(test_label, pred_keras)

        # Compute AUPRC
        auprc = auc(recall, precision)

        # Plot the PR curve
        plt.figure(3)
        plt.plot(recall, precision, label=f"AUPRC = {auprc:.3f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend(loc="lower left")
        if save:
            fig_save_name = os.path.join(
                self.save_path, f"precision_recall_curve_{name}.png"
            )
            plt.savefig(fig_save_name)
        else:
            plt.show()

    def evaluate(self, model, test_data, test_label, group_test):
        preds = model.predict(test_data)
        pred_keras = model.predict(test_data)[:, 1]
        class_preds = np.argmax(preds, axis=1)

        pred_data = pd.DataFrame()
        pred_data["y_pred"] = class_preds
        pred_data["y_proba"] = pred_keras
        pred_data["y_true"] = test_label
        pred_data["group"] = group_test

        group_data = (
            pred_data.groupby("group")
            .agg(
                {
                    "y_pred": lambda x: x.value_counts().index[0],
                    "y_proba": lambda x: x.mean(),
                    "y_true": lambda x: x.value_counts().index[0],
                }
            )
            .reset_index()
        )

        for type, df in {"all": pred_data, "group": group_data}.items():
            self._logger.info(f"Results for {type} : ")

            self.report = classification_report(
                df["y_true"], df["y_pred"], output_dict=True
            )
            self.confusion_matrix = confusion_matrix(df["y_true"], df["y_pred"])

            f1 = f1_score(df["y_true"], df["y_pred"], average="weighted")
            # ROC_AUC = roc_auc_score(to_categorical(df['y_true']), preds, average=None)
            self._logger.info(
                f"Accuracy sklearn is : {accuracy_score(df['y_true'], df['y_pred'])}"
            )

            self._logger.info(f"f1 score is : {f1}")

            if type == "all":
                score = model.evaluate(
                    test_data, to_categorical(df["y_true"]), verbose=0
                )
                self._logger.info(f"Test accuracy keras  is : {score[2]}")
                self._logger.info(f"Test loss  is : {score[0]}")
            else:
                self._logger.info(
                    f"Test accuracy  is : {accuracy_score(df['y_true'], df['y_pred'])}"
                )

            fpr_keras, tpr_keras, thresholds_keras = roc_curve(
                df["y_true"], df["y_proba"]
            )
            auc_keras = auc(fpr_keras, tpr_keras)

            # self._logger.info(f"Test AUC keras  is : {score[1]}")
            self._logger.info(f"Test AUC sklearn  is : {auc_keras}")
            # self._logger.info(f"Test ROC AUC sklearn  is : {ROC_AUC}")

            self.plot_roc_auc_curves(df["y_proba"], df["y_true"], save=True, name=type)
            self.plot_micro_roc_auc_curves(
                df["y_proba"], df["y_true"], save=True, name=type
            )
            self.plot_pr_curves(df["y_proba"], df["y_true"], save=True, name=type)

    def save(self):
        """
        Save the results associated with model evaluation : classification report, learning curves and confusion matrix
        """

        df = pd.DataFrame(self.report).transpose()
        df.to_csv(os.path.join(self.save_path, "result.csv"))
        self._logger.info(f"Classification report saved in {self.save_path}")

        # saving confusion Matrix
        fig = plt.figure()
        sns.heatmap(self.confusion_matrix, annot=True)
        fig.savefig(os.path.join(self.save_path, "confusion_matrix.png"))
        self._logger.info(f"Confusion Matrix saved in {self.save_path}")

        # saving learning curves
        fig = self.plot_history()
        fig.savefig(os.path.join(self.save_path, "learning_curves.png"))
        self._logger.info(f"Learning curves saved in {self.save_path}")
