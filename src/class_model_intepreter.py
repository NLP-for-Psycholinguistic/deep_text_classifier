"""
creation_date: 4 august 2022
author : Quillivic Robin

Description : 
- Define ModelTrainer
"""

from distutils import text_file
import os, sys
import logging
import numpy as np
import pandas as pd

from lime.lime_text import LimeTextExplainer


class LimeModelInterpreter:
    def __init__(self, config, preprocessor, model, label_list) -> None:
        self.config = config
        self._logger = logging.getLogger(self.__class__.__name__)

        self.preprocessor = preprocessor
        self.model = model
        self.class_names = label_list
        self.split_expression = " "
        self.explainer = LimeTextExplainer(
            class_names=self.class_names,
            split_expression=self.split_expression,
            bow=False,
        )

    def classifier(self, text_list: list) -> list:
        """Compute a prediction from raw_text

        Args:
            text (str): _description_

        Returns:
            list: _description_
        """
        sequence = self.preprocessor(text_list)
        pred = self.model.predict(sequence)
        return pred

    def analyse(self, text_list: list, num_features=20) -> list:
        """
        Make the interpretation from th emodel for a list of text

        Args:
            text_list (list): _description_

        Returns:
            list: liste of results
        """

        results = []  # all explaner

        # Create an explainer for each text
        for t, text in enumerate(text_list):
            try:
                results += [
                    self.explainer.explain_instance(
                        text,
                        self.classifier,
                        num_features=num_features,
                        labels=self.class_names,
                    )
                ]
            except Exception as e:
                self._logger.warning(
                    f"Fail to compute the interpretation {t} because of {e}"
                )
        self.results = results
        self._logger.info("All the local interpretation were done with succes")

    def compute_global_analysis(self):
        """_summary_

        Args:
            results (_type_): _description_
        """
        classes_explainers = []
        for classe in self.class_names:
            explainer_dict = {}
            for i in range(len(self.results)):
                try:
                    for word, score in self.results[i].as_list(classe):
                        explainer_dict[word] = explainer_dict.get(word, []) + [score]
                except Exception as e:
                    self._logger.info(f"Adding word failed because of {e}")
                    continue
            average_dict = {}
            for elt in explainer_dict.keys():
                average_dict[elt] = np.mean(explainer_dict[elt])

            average_dict = {
                k: v
                for k, v in sorted(
                    average_dict.items(), key=lambda item: item[1], reverse=True
                )
            }
            classes_explainers.append(average_dict)

        self.classes_explainers = classes_explainers
        self._logger.info("The global interpretation was terminated with succes")

    def save(self, saving_folder):
        """
        Save an example  interpretation in html format in the saving folder
        Save the orderd word importance dict for each class

        Args:
            saving_folder (_type_): _description_
        """
        for i in range(len(self.classes_explainers)):
            interpretation = pd.DataFrame.from_dict(
                self.classes_explainers[i], orient="index", columns=["score"]
            )
            saving_path = os.path.join(saving_folder, f"interpretation_class_{i}.csv")
            interpretation.to_csv(saving_path)
        self._logger.info("The global interpretation was successfully saved !")

        if not os.path.exists(os.path.join(saving_folder, "htmls")):
            os.mkdir(os.path.join(saving_folder, "htmls"))

        for i, exp in enumerate(self.results):
            html = exp.as_html()
            open(os.path.join(saving_folder, "htmls", f"{i}.html"), "w").write(html)
        self._logger.info("The html files interpretation were successfully saved !")
