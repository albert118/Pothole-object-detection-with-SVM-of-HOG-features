# Dataset

This project is comprised of two datasets. The [biankatpas dataset](https://github.com/biankatpas/Cracks-and-Potholes-in-Road-Images-Dataset) as well as the [Kaggle set](https://www.kaggle.com/datasets/sovitrath/road-pothole-images-for-pothole-detection?resource=download).

The two raw datasets are held here under their named directories. 

The compiled dataset is saved as a `.csv` in the src directory during script execution.

[Kaggle dataset](https://www.kaggle.com/datasets/sovitrath/road-pothole-images-for-pothole-detection?select=PotholeDataset.pdf )

* 4409 records

[GitHub dataset from the portugues project](https://github.com/biankatpas/Cracks-and-Potholes-in-Road-Images-Dataset)

* 1235 records 

## Set Up your Local Copy

Given the size of the two datasets, I've avoided re-publishing them to this repo (the Kaggle dataset is 9GB+). So their is some set up to get your local data ready.

First set up the larger of the two datasets.

1. Download the Kaggle data set as a zip
2. Extract and rename the `Dataset 1 (Simplex)` folder into the Kaggle directory.

Second, set up the biankatpas dataset,

1. clone the git repo from [biankatpas](https://github.com/biankatpas/Cracks-and-Potholes-in-Road-Images-Dataset).
2. Copy the `Dataset` folder into the biankatpas directory.

## Compile the dataset

In the interest of time, I didn't modularise the pipeline into a CLI script. So you will have to modify the [`runner.py`](/src/runner.py) script as needed.

1. Uncomment the `create_dataset` method, `fn`, `logger`, and `to_csv` lines. This will enable loading then saving the compiled dataset to disk.
  * optionally, enable "preview" to `True` to verify the HOG descriptor extraction process.
  * also consider modifying "n_jobs", 16 threads will load your CPU to 100% on a desktop easily. Lower is recommended for laptops
2. Done! Validate the CSV file is to your liking

