import logging
from os.path import splitext
from pathlib import Path

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

class FileScanner:
    def get_potential_files(self, scan_dir, regex: str, search_subdirs: bool):
        potential_files = []

        if search_subdirs:
            subdirs = [subdir for subdir in Path(Path(__file__).parent, scan_dir).iterdir() if subdir.is_dir()]

            for subdir in subdirs:
                [potential_files.append(file) for file in subdir.rglob(regex) if file.is_file()]
        else:
            for file in Path(scan_dir).rglob(regex):
                if file.is_file(): potential_files.append(file) 

        return potential_files


def scan_directory(scan_dir: str, reg_pattern: str, search_subdirs: bool=True):
    return FileScanner().get_potential_files(scan_dir, reg_pattern, search_subdirs=search_subdirs)
