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
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA, KernelPCA
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier


def main(cwd_path: Path, logger: logging.Logger) -> None:
    logger.info("-------------------------------------------------------------------------")
    # region init
    test_data_path = cwd_path / Path("test.csv")
    train_data_path = cwd_path / Path("train.csv")
    pairplot_file = "pairplot.png"
    random_state = 42
    # endregion 

    # region reading from file
    logger.info("reding training file")
    df_train = pd.read_csv(train_data_path)
    df_train["class"] = df_train["class"].map({"GALAXY": 0, "STAR": 1, "QSO": 2})
    logger.info("remapping of class data {\"GALAXY\": 0, \"STAR\": 1, \"QSO\": 2}")
    logger.info("reding test file")
    df_test = pd.read_csv(test_data_path)

    # df_nan_values = df_train[df_train.isna().any(axis=1)]
    # print(len(df_nan_values))
    # endregion 

    # region pairplot
    if os.path.isfile(cwd_path / Path(pairplot_file)):
        logger.warning("%s file exist skipping pair plot", pairplot_file)
    else:
        logger.info("plotting pairplot")
        pairplot = sns.pairplot(df_train.sample(100), hue="class", height=2.0)
        logger.info("saving plot to file %s", pairplot_file)
        pairplot.savefig(cwd_path / Path("pairplot.png"))
        plt.clf()
    # endregion 

    # region exstracting data as nupy array
    logger.info("converting to numpy array")

    train_np_array = df_train.dropna().drop("class", axis=1).to_numpy()
    train_class_np_array = df_train.dropna()["class"].to_numpy()
    test_np_array = df_test.dropna().to_numpy()
    # endregion 

    # region spliting data for training
    logger.info("splitting X_train_standardized for training")

    train_test_split =  model_selection.train_test_split(
        train_np_array, 
        train_class_np_array, 
        test_size=0.8, 
        random_state=random_state
    )

    x_train, x_test, y_train, y_test =  train_test_split
    # endregion 

    # region setting upp a pipeline
    logger.info("creating pipeline")
    pipeline = Pipeline([
        ('scaling',             StandardScaler()),
        ('preprocessoer_pca',   PCA(n_components=4)),
        ('svc',          svm.SVC())
    ])
    # endregion 

    # region setting upp grid search
    logger.info("setting upp parameters")
    param_range_C = np.logspace(-3,3,2).tolist()
    param_range_gamma = np.logspace(-3,2,2).tolist()

    param_grid = [
        {
            'svc__C': param_range_C,
            'svc__kernel': ['linear']
        },
        {
            'svc__C': param_range_C,
            'svc__kernel': ['rbf'],
            'svc__gamma': param_range_gamma
        }
    ]

    logger.info("defining grid search using param_grid")
    grid_search_cv = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=2,
        n_jobs=-1,
        verbose=3
    )

    logger.info("fitting grid search to training set. this might take a long time")
    grid_search_cv.fit(x_train, y_train)
    best_score = str(grid_search_cv.best_score_)
    best_params = str(grid_search_cv.best_params_)
    # logger.info("best score was $s", best_score)
    # logger.info("best parameters was $s", best_params)
    # endregion

    # region running pipeline
    # # Fit the pipeline on training data
    # logger.info("fitting pippeline to training split")
    # pipeline.fit(x_train, y_train)

    # # Make predictions
    # predictions = pipeline.predict(x_test)

    # # Evaluate the model
    # train_accuracy = pipeline.score(x_train, y_train)
    # print(f"train_accuracy: {train_accuracy}")
    # test_accuracy  = pipeline.score(x_test, y_test)
    # print(f"test_accuracy: {test_accuracy}")
    # endregion 



def logger_init(cwd_path: Path, logger_level, toterm: bool = False) -> logging.Logger:
    """
    # initialises a global logger for this program

    ## Args
    - cwd_path (Path): 
       Path to the directory to save the logg file
    - logger_level (logging._Logger): 
        Level use for logging DEBUG, INFO, WARNING, ERROR
    - toterm (bool): 
        Enabels logging to terminal

    # Returns
    - Logger (logging.logger):
        the logger ogject to use for logging
    """
    logger_file = "logg.txt"
    logger_path = cwd_path / Path(logger_file)

    # if os.path.isfile(logger_path):
    #     os.remove(logger_path)

    logger = logging.getLogger(__name__)

    if toterm:
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
    else:
        logging.basicConfig(
            filename=logger_path,
            encoding='utf-8', 
            level=logger_level,
            format='%(asctime)s - %(levelname)-7s: %(message)s ',
            datefmt='%m/%d/%Y %I:%M:%S',
        )

    return logger

if __name__ == "__main__":
    cwd_path = Path(__file__).parent
    logger = logger_init(cwd_path, logging.INFO, toterm=True)
    main(cwd_path, logger)