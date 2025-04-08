#!/usr/bin/env python3
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import os
from sklearn import svm
from sklearn.linear_model import Perceptron, LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA

from sklearn.neighbors import KNeighborsClassifier


def main(cwd_path: Path, logger: logging.Logger) -> None:

    ## init
    test_data_path = cwd_path / Path("test.csv")
    train_data_path = cwd_path / Path("train.csv")
    pairplot_file = "pairplot.png"

    # reading from file
    logger.info("reding training file")
    df_train = pd.read_csv(train_data_path)
    logger.info("reding test file")
    df_test = pd.read_csv(test_data_path)

    # pairplot
    if os.path.isfile(cwd_path / Path(pairplot_file)):
        logger.warning("%s file exist skipping pair plot", pairplot_file)
    else:
        logger.info("plotting pairplot")
        pairplot = sns.pairplot(df_train.sample(100), hue="class", height=2.0)
        logger.info("saving plot to file %s", pairplot_file)
        pairplot.savefig(cwd_path / Path("pairplot.png"))
        plt.clf()


    # exstracting data as nupy array
    logger.info("converting to numpy array")
    train_np_array = df_train.dropna().drop("class", axis=1).to_numpy()
    test_np_array = df_test.dropna().to_numpy()

    # normalizing data 
    logger.info("normalizing data")
    standard_scaler = StandardScaler()
    X_train_standardized = standard_scaler.fit_transform(train_np_array[:,:17])
    X_test_standardized = standard_scaler.transform(test_np_array[:,:17])




def logger_init(cwd_path: Path, logger_level) -> logging.Logger:
    logger_file = "logg.txt"
    logger_path = cwd_path / Path(logger_file)

    # if os.path.isfile(logger_path):
    #     os.remove(logger_path)

    logger = logging.getLogger(__name__)

    logging.basicConfig(
        encoding='utf-8', 
        level=logger_level,
        format='%(asctime)s - %(levelname)-7s: %(message)s ',
        datefmt='%m/%d/%Y %I:%M:%S',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(logger_path)
        ],
    )

    return logger


if __name__ == "__main__":
    cwd_path = Path(__file__).parent
    logger = logger_init(cwd_path, logging.INFO)
    main(cwd_path, logger)