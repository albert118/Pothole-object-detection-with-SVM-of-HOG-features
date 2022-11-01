from multiprocessing import Pool

import pandas as pd

from extractor import Extractor as e
from feature_extractor import hog_extractor as hg

config = {
    "blur_sigma": 3,                     # blur "intensity" of images
    "blur_truncate": 2.5,                # blur region size
    "scale_factor": 125,                 # downscale (or up) images in fixed aspect ratio (pixels)
    "dataset_limit": 10,                 # limit the dataset size utilised
    "preview": False,                    # preview the HOG images
    "dataset_fn": "pothole_dataset.csv", # name of the resultant dataset file
    "hog_orientations": 4,               # HOG param
    "hog_pixels_per_cells": 16,          # HOG param
    "hog_cells_per_block": 2             # HOG param
}

def run(pool):
    raw = e.extract_training_data()[:config["dataset_limit"]]

    if config['preview']: 
        hg.display_extracted_hog_descriptors(raw.resource_name, config)
    else:
        descriptors = pd.DataFrame(hg.extract_hog_descriptors(raw.resource_name, pool, config))
        raw = pd.concat([raw, descriptors], axis=1)
    
    return raw


if  __name__ == '__main__':
    n = 10
    with Pool(n) as pool:
        raw = run(pool)

    raw.to_csv(config["dataset_fn"])

