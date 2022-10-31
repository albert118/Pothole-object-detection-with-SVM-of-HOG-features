import logging
from os.path import splitext
from pathlib import Path

from .Validator import validate

_logger = logging.getLogger(__name__)

# |- biankatpas
#    |
#    |- 12345_67890_12345
#    |   |- 12345_67890_12345_RAW.jpg
#    |- 12345_67890_12345
#        |- 12345_67890_12345_RAW.jpg   


# |- kaggle
#    |
#    |- Dataset
#    |   |- Train
#    |   |   |- 123456789.JPG
#    |   |- Test
#    |   |   |- Negative data
#    |   |   |   |- 123456789.JPG
#    |   |   |- Positive data
#    |   |   |   |- 123456789.JPG




biankatpas_fn = "data/biankatpas"
kaggle_fn = "kaggle/Dataset"

class FileScanner:
    def get_potential_files(self, scan_dir, regex: str, search_subdirs: bool=True):
        potential_files = []

        if search_subdirs:
            subdirs = [subdir for subdir in Path(scan_dir)]

            for subdir in subdirs:
                [potential_files.append(file) for file in subdir.rglob(regex) if file.is_file()]
        else:
            for file in Path(scan_dir).rglob(regex):
                if file.is_file(): potential_files.append(file) 

        return potential_files
    
    def pre_process(self, fn):
        if (not validate(fn)):
            message = f"couldn't validate the file name: '{fn}', no moves were made"
            _logger.error(message)
            raise ValueError(message)


def scan_directory(scan_dir: str, reg_pattern: str):
    scanner = FileScanner()
    potential_files = scanner.get_potential_files(scan_dir, reg_pattern)

    for file in potential_files: scanner.pre_process(file)

    return potential_files
