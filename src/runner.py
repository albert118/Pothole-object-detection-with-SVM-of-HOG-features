from extractor import Extractor as e
from feature_extractor import hog_extractor as hg

df = e.extract_training_data()


config = {
    "blur_sigma": 2.5,
    "blur_truncate": 2.5,
    "scale_factor": 150
}

hg.extract_hog_descriptors(df.resource_name, config)
