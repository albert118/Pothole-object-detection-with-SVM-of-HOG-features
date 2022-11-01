import logging
from pathlib import Path

import pandas as pd

from .FileScanner import scan_directory

_logger = logging.getLogger(__name__)

src_dir = Path(Path(Path(__file__).parent).parent).parent


class Extractor:
    def __init__(self, **kwargs):
        kwargs.setdefault('biankatpas_limit', 100)

        [self.__setattr__(key, kwargs.get(key)) for key in kwargs]

        self.merged_frame = None
        self._sets = []

    def open_biankatpas(self, scan_dir):
        image_resources = scan_directory(scan_dir, "*_RAW.jpg")
        self.biankatpas_df = pd.DataFrame(image_resources, columns=["resource_name"])

        # limit the content
        if (self.biankatpas_limit != 0): self.biankatpas_df = self.biankatpas_df[:self.biankatpas_limit]

        self._sets.append(self.biankatpas_df)
        self.biankatpas_df["dataset"] = "Biankatpas"

        return self

    def open_kaggle_train(self, scan_dir):
        true_image_resources = scan_directory(Path(scan_dir, 'Positive data'), "*.JPG", search_subdirs=False)
        self.kaggle_df_true = pd.DataFrame(true_image_resources, columns=["resource_name"])

        self._sets.append(self.kaggle_df_true)
        self.kaggle_df_true["dataset"] = "Kaggle"

        false_image_resources = scan_directory(Path(scan_dir, 'Negative data'), "*.JPG", search_subdirs=False)
        self.kaggle_df_false = pd.DataFrame(false_image_resources, columns=["resource_name"])

        self._sets.append(self.kaggle_df_false)
        self.kaggle_df_false["dataset"] = "Kaggle"

        return self

    def open_kaggle_test(self, scan_dir):
        true_image_resources = scan_directory(scan_dir, "*.JPG", search_subdirs=False)
        self.kaggle_df = pd.DataFrame(true_image_resources, columns=["resource_name"])

        self._sets.append(self.kaggle_df)
        self.kaggle_df["dataset"] = "Kaggle"

        return self

    def add_training_labels(self):
        self.biankatpas_df["class"] = True
        self.kaggle_df_true["class"] = True
        self.kaggle_df_false["class"] = False
        return self

    def merge_sets(self):
        self._merged = pd.concat(self._sets)
        return self

    def extract(self): return self._merged

####################################################################
#    ID   #    resource_name    #   dataset   # class (True/False) #
####################################################################

def extract_training_data(config: dict):
    biankatpas_fn = "data/biankatpas"
    kaggle_fn = "data/kaggle/Dataset/Train data"
    # notably, train_df.csv content is excluded here
    # training with a pre-counted number of potholes seems counter productive...

    try:
        return (
            Extractor(**config)
                .open_biankatpas(Path(src_dir, biankatpas_fn))
                .open_kaggle_train(Path(src_dir, kaggle_fn), )
                .add_training_labels()
                .merge_sets()
                .extract()
                # add a sensible index without polluting the columns
                .reset_index(drop=True)
        )
    except Exception as ex:
        _logger.error(f"error while extracting the data '{ex}'")
        raise

#############################################
#    ID   #    resource_name    #   dataset #
#############################################

def extract_testing_data(config: dict):
    biankatpas_fn = "data/biankatpas"
    kaggle_fn = "data/kaggle/Dataset/Test data"
    limit_testing_size = 300

    # TODO include the simpleTestFullSizeAllPotholesSortedFullAnnotation.txt annotation content

    try:
        return (
            Extractor()
                .open_biankatpas(Path(src_dir, biankatpas_fn))
                .open_kaggle_test(Path(src_dir, kaggle_fn))
                .merge_sets()
                .extract()
                # add a sensible index without polluting the columns
                .reset_index(drop=True)
        )
    except Exception as ex:
        _logger.error(f"error while extracting the data '{ex}'")
        raise
