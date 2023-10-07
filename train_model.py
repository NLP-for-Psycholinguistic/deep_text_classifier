"""

creation_date: 4 august 2022
author : Quillivic Robin

Description : 
- Script to execute to, train, save evaluate and interpret a model

"""



import csv
import random
import pandas as pd
import numpy as np

import logging
import logging.config
import os, sys
import yaml
import shutil

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc




import tensorflow as tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  

from src import utils
from src.class_data_loader import DataLoader
from src.class_preprocessing import DataPreprocessor
from src.class_model_builder import ModelBuilder
from src.class_model_trainer import ModelTrainer
from src.class_model_intepreter import LimeModelInterpreter


with open(os.path.join(os.path.dirname(__file__), 'logging_config.yaml'), 'r') as stream:
    log_config = yaml.load(stream, Loader=yaml.FullLoader)
logging.config.dictConfig(log_config)
logger = logging.getLogger('training')


config_filename = "config.yaml"


def get_gpu_info(logger):
    logger.info(f"You are using tf version : {tf.version.VERSION}")
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus :
        logger.info("GPU is available")
        try:
        # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.list_logical_devices('GPU')
                logger.info(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
        except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
            logger.error(e)
    else :
        logger.info("GPU is NOT AVAILABLE")

def get_saving_folder(config, logger):
    saving_dir = config['results']['folder']
    if config['results']['name']=="":
        filters = "_".join(config['data']['filter']['pos_filter']) + f"dys_drop {str(config['data']['filter']['drop_dysfluences'])}"
        subgroups = '_'.join(config['data']['filter']['subgroup'][1])
        cut = str( config['data']['filter']['max_number_word'])
        training_name = f"{cut}_{subgroups}_{config['data']['filter']['target']}_{config['model']['kind']}_{config['data']['sequence']['size']}_{filters}"
    else :
        training_name = config['results']['name']
    saving_folder = os.path.join(saving_dir, training_name)
    if not os.path.exists(saving_folder):
        os.mkdir(saving_folder)
    logger.info(f'The config file used is {config_filename}')
    logger.info(f'Models and results will be saved in {saving_folder}')

    return saving_folder


def load_data(config, saving_folder):
     with tf.device('/cpu:0'):
        loader = DataLoader(config)
        data = loader.data
        train, test, valid = loader.get_train_test_valid()
        label_list = np.unique(train['label'].values)
        loader.save_data_description(saving_folder)
        return train, test, valid, label_list

def preprocess_data(config, train, test, valid):
    preprocessor = DataPreprocessor(config)
    if config['model']["kind"] in ["lstm", 'cnn', "transformer", "attention", "custom"]:
        train_data, train_label,train_group, test_data, test_label, test_group, valid_data, valid_label, valid_group = preprocessor.prepare_data_for_keras_model(train, test, valid)
        print(np.unique(train_label))
    elif config['model']["kind"] in ['distilbert_multi_cased', "labse"] :
        train_data, train_label, test_data, test_label, valid_data, valid_label = preprocessor.prepare_data_for_bert_model(train, test, valid)
    elif config['model']['kind'] in ["camembert", "flaubert"] :
        train_data, train_label, test_data, test_label, valid_data, valid_label = preprocessor.prepare_data_for_camembert_model(train, test, valid)
    else :
        raise NotImplementedError

    return  train_data, train_label,train_group, test_data, test_label, test_group, valid_data, valid_label, valid_group, preprocessor

def build_model(config, label_list, preprocessor) :
    model_builder = ModelBuilder(config, label_list=label_list, seq_max_length = preprocessor.max_length)
    model = model_builder.model

    return model

def train_model(config, saving_folder, model, train_data, train_label, train_group,
                valid_data, valid_label, valid_group, 
                test_data, test_label, test_group):
    trainer = ModelTrainer(config, saving_folder)
    fitted_model = trainer.train(model, train_data, train_label, valid_data, valid_label)
    trainer.evaluate(fitted_model, test_data, test_label, test_group)
    trainer.save()

    return fitted_model

def interpret_model(config, preprocessor, model, label_list, test, saving_folder):
    if config['model']["kind"] in ["lstm", 'cnn', "transformer", "attention", "custom"]:
        preprocessing_function = preprocessor.keras_preprocessor
    elif config['model']["kind"] in ['distilbert_multi_cased', "labse"] :
        preprocessing_function = preprocessor.bert_preprocessor
    elif config['model']['kind'] in ["camembert", "flaubert"] :
        preprocessing_function = preprocessor.camembert_preprocessor
    else :
        raise NotImplementedError

    interpreter = LimeModelInterpreter(config,preprocessor = preprocessing_function, model = model, label_list=label_list)
    interpreter.analyse(test.sample(config['interpreter']['num_samples'])['text'].tolist(), num_features=config['interpreter']['lime']['num_features'])
    interpreter.compute_global_analysis()
    interpreter.save(saving_folder)


def main_one_model(config, logger):
    logger.info('#'*10)
    logger.info('#GPU information and memory configuration')
    logger.info('#'*10)

    get_gpu_info(logger)

    logger.info('#Clearing memory information')
    tf.keras.backend.clear_session()

    logger.info('#'*10)
    logger.info('#Configuration of saving')
    logger.info('#'*10)

    saving_folder = get_saving_folder(config, logger)

    logger.info('#'*10)
    logger.info('#Data loading')
    logger.info('#'*10)

    train, test, valid, label_list = load_data(config, saving_folder=saving_folder)
    for name, data in {"train": train, "test":test, "valid":valid}.items():
        data.to_csv(os.path.join(saving_folder,name+".csv"))

    logger.info('#'*10)
    logger.info('#Data preprocessing')
    logger.info('#'*10)

    train_data, train_label,train_group, test_data, test_label, test_group, valid_data, valid_label, valid_group, preprocessor = preprocess_data(config, train, test, valid)
    
    logger.info('#'*10)
    logger.info('#Model Building')
    logger.info('#'*10)
    
    model = build_model(config, label_list, preprocessor)

    logger.info('#'*10)
    logger.info('#Model Training, Evaluation, Saving')
    logger.info('#'*10)

    fitted_model = train_model(config, saving_folder, model, train_data, train_label, train_group,
                valid_data, valid_label, valid_group, 
                test_data, test_label, test_group)
    
    logger.info('#'*10)
    logger.info('#Model Intepreting')
    logger.info('#'*10)
    
    #interpret_model(config, preprocessor, fitted_model, label_list, test = test, saving_folder = saving_folder)

    logger.info('#'*10)
    shutil.copyfile(config_filename, os.path.join(saving_folder,"config.yaml"))
    logger.info(f'Done and saved at {saving_folder}')
    logger.info('#'*10)

    return  train_data, train_label, train_group, test_data, test_label, test_group, valid_data, valid_label, valid_group, fitted_model



    

def compute_average_roc_curve(config, logger,N_iter = 100):
    tprs, aucs = [], []
    tprs_group, aucs_group = [], []
    mean_fpr = np.linspace(0, 1, N_iter)
    mean_fpr_group = np.linspace(0, 1, N_iter)
    sns.set(font_scale=2)
    for seed in random.sample(range(0, 1000), N_iter) :
    #for seed in [447,985, 47]:
        config['data']['filter']['seed'] = seed
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        train_data, train_label,train_group, test_data, test_label, test_group, valid_data, valid_label, valid_group, model = main_one_model(config, logger)

        pred_keras = model.predict(test_data)[:,1]

        preds = model.predict(test_data)
        class_preds = np.argmax(preds, axis=1)
        
        pred_data = pd.DataFrame()
        pred_data["y_pred"] = class_preds
        pred_data['y_proba'] = pred_keras
        pred_data["y_true"] = test_label
        pred_data["group"] = test_group

        group_data = pred_data.groupby("group").agg({"y_pred": lambda x: x.value_counts().index[0],
                                                     "y_proba":lambda x: x.mean() ,
                                                     "y_true": lambda x: x.value_counts().index[0]}).reset_index()

        
        #print("PREDICTIONS: ", pred_keras)
        fpr, tpr, thresholds = roc_curve(test_label, pred_keras, sample_weight = [1]*len(test_label), pos_label = config["model"]['pos_label'])
        fpr_group, tpr_group, thresholds = roc_curve(group_data['y_true'], group_data['y_proba'], pos_label = config["model"]['pos_label'])
        
        auc_keras = auc(fpr, tpr)
        auc_keras_group = auc(fpr_group, tpr_group)
        
        logger.info(f'Save_AUC is : {auc_keras}, associated with {seed}')
        logger.info(f'Save_AUC for GROUP is : {auc_keras_group}, associated with {seed}')

        interp_tpr = np.interp(mean_fpr,fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(auc_keras)

        interp_tpr_group = np.interp(mean_fpr_group,fpr_group, tpr_group)
        interp_tpr_group[0] = 0.0
        tprs_group.append(interp_tpr_group)
        aucs_group.append(auc_keras_group)

    def plot_curves(tprs, aucs, mean_fpr,name = ""):
        _, ax = plt.subplots(figsize=(10, 10))
        ax.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        
        ax.plot(
            mean_fpr,
            mean_tpr,
            label=fr"Mean ROC {config['model']['kind']} (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
            lw=2,
            alpha=0.8,
            )   

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    
        ax.fill_between(
            mean_fpr,
            tprs_lower,
            tprs_upper,
            alpha=0.2,
            label=fr" {config['model']['kind']}$\pm$ 1 std.",
            )

        ax.set(
            xlim=[-0.05, 1.05],
            ylim=[-0.05, 1.05],
            xlabel="False Positive Rate",
            ylabel="True Positive Rate",
            title=f"Mean ROC curve with variability\n(Positive label {config['data']['filter']['target']}, {N_iter} runs)",
        )
        ax.axis("square")
        ax.legend(loc="lower right")

        if config['model']["focal_loss"]:
            saving_path = os.path.join(config['results']['folder'],f"{config['model']['kind']}_{config['data']['filter']['target']}_average_roc_auc_{config['data']['sequence']['size']}_focal_{name}.jpeg")
        else :
            saving_path = os.path.join(config['results']['folder'],f"{config['model']['kind']}_{config['data']['filter']['target']}_average_roc_auc_{config['data']['sequence']['size']}_{name}.jpeg")

        plt.savefig(saving_path,bbox_inches='tight', dpi = 200)

        # empty the memory (ax.clear() does not work)
        fig, ax = plt.subplots(figsize=(10, 10))
        ax = None
        fig= None
    
    plot_curves(tprs, aucs, mean_fpr,name = "")

    plot_curves(tprs_group, aucs_group, mean_fpr_group,name = "group")

if __name__ == "__main__":
 
    config = utils.load_config(config_filename)
    #train, save and intepret one model
    try : 
        N_tries = int(sys.argv[1])
    except :
        N_tries = 10
    
    
    # "full_or_partial_PTSD", "CC_probable","CD_probable","CC_probable", "CD_probable", "full_or_partial_PTSD","CB_probable", "CC_probable", "CD_probable"
    for target in ["full_or_partial_PTSD"]:
        config['data']['filter']['target'] = target

        logger.info('#'*10)
        logger.info(f"Target is {target}")
        logger.info('#'*10)

        compute_average_roc_curve(config, logger, N_tries)


  



    






