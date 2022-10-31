import logging

import pandas as pd

from FileScanner import scan_directory

_logger = logging.getLogger(__name__)


class Extractor:
    def __init__(self):
        self.merged_frame = None

    def open_biankatpas(self, scan_dir):
        image_resources = scan_directory(scan_dir, "*_RAW.jpg")
        self.biankatpas_df = pd.DataFrame(image_resources, columns=["resource_name"])
        return self

    def open_kaggle(self):
        image_resources = scan_directory(scan_dir, "*.JPG", search_subdirs=False)
        self.kaggle_df = pd.DataFrame(image_resources, columns=["resource_name"])
        return self

    def merge_sets(self):
        self._merged = pd.concat([
            self.biankatpas_df,
            self.open_kaggle
        ]).reset_index()

        return self

    def extract(self): return self._merged

######################
#    ID   #    fn    #
######################

def extract():
    try:
        return (
            Extractor()
                ## Grab the initial sets
                .open_biankatpas()
                .open_kaggle()
                # Merge them
                .merge_sets()
                # Get the data
                .extract()
        )
    except Exception as ex:
        _logger.error(f"error while extracting the data '{ex}'")
        raise
