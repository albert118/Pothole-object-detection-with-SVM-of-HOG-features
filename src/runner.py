import logging
from multiprocessing import Pool

import pandas as pd
from sklearn.model_selection import train_test_split

from extractor import Extractor as e
from feature_extractor import hog_extractor as hg
from svm.model import Model

LOG_LVL = logging.INFO

def setup_logger():
	logging.basicConfig(level=LOG_LVL)
	return logging.getLogger(__name__)


config = {
    "blur_sigma": 3,                     # blur "intensity" of images
    "blur_truncate": 2.5,                # blur region size
    "scale_factor": 125,                 # downscale (or up) images in fixed aspect ratio (pixels)
    "dataset_limit": None,               # limit the dataset size utilised
    "biankatpas_limit": 500,             # limit the impact of the large positive-only dataset (biankatpas)
    "preview": False,                    # preview the HOG images
    "dataset_fn": "pothole_dataset.csv", # name of the resultant dataset file
    "hog_orientations": 4,               # HOG param
    "hog_pixels_per_cells": 16,          # HOG param
    "hog_cells_per_block": 2,            # HOG param
    "n_jobs": 16,                        # number of job pools to create (multi threading)
}


def load_dataset(logger, pool, config): 
    # bug with creating then loading adds the index col 'Unnamed: 0'
    # def load_dataset(pool, config): return pd.read_csv(config["dataset_fn"], index_col="Unnamed: 0")
    
    logger.info("loading up the dataset from disk")
    return pd.read_csv(config["dataset_fn"])

def create_dataset(pool, config):
    if config["dataset_limit"]:
        raw = e.extract_training_data(config)[:config["dataset_limit"]]
    else:
        raw = e.extract_training_data(config)

    if config['preview']: 
        hg.display_extracted_hog_descriptors(raw.resource_name, config)
    else:
        descriptors = pd.DataFrame(hg.extract_hog_descriptors(raw.resource_name, pool, config))
        raw = pd.concat([raw, descriptors], axis=1)
    
    return raw

def train_and_eval(logger, dataset, config):
    model = Model(**config)

    # use .values here to avoid passing the df column names
    # avoids "X does not have valid feature names, but StandardScaler was fitted with feature names"
    X = dataset.drop(columns=["resource_name", "class", "dataset"]).values
    y = dataset["class"].values

    x_train, x_test, y_train, y_test = train_test_split(X, y)

    model.train(x_train, y_train)

    score = model.model.score(x_test, y_test)
    logger.info(f"score of {score*100}%")

def run(logger, pool, config):
    # raw = create_dataset(pool, config)
    raw = load_dataset(logger, pool, config)

    # fn = config["dataset_fn"]
    # logger.info(f"writing the dataset to disk '{fn}'")
    # raw.to_csv(fn)

    logger.info("training and evaluated an SVM (hold tight)")
    train_and_eval(logger, raw, config)

if  __name__ == '__main__':
    logger = setup_logger()

    n = config["n_jobs"]
    with Pool(n) as pool:
        logger.info("beggining pipeline")
        run(logger, pool, config)
