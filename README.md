# Image Processing and Pattern Recognition (2022 Group 27)

2022 Spring Session UTS Image Processing Project Repository. 

# Project Description

Aim: develop a pothole object detection system.

The main function of this project is to automatically detect cracks on the road. 

This project can be used in road maintenance to greatly improve the efficiency of finding cracks. 

Similarly, this technology can also be used in future autonomous driving to improve the safety of future traffic. 

# Set Up

## Set Up Python Environment for Development

Using Python verison 3.9.+ to 3.10.0, set up the environment like so.

Create and source your choice of virtual environment, eg. [venv](https://virtualenv.pypa.io/en/latest/). I also recommend the [win wrapper](https://pypi.org/project/virtualenvwrapper-win/) if you're on Windows. Activate it, then install the dependencies with pip like so,

```
// once in your development environ
pip install -r requirements
```

## Set Up the Dependent Datasets

Two dependent datasets must be compiled into a single source before running the pipeline. They are sourced from Kaggle and GitHub. See the data [README](/data/README.md)

## Running the script

After activating your virtualenv and installing dependencies, run the following,

```
cd src
python runner.py
```

This will run the pipeline and graph some results. The accuracy will be logged to command line.

# Guides

## Using GitHub / Git

As is standard, the `master` branch is protected and cannot be pushed to directly. You will receive an error if you try.
Checking out a local development -> then squash merging to master is the process that ensures everyone sees and understands the new changes.

The [GitHub desktop app](https://desktop.github.com/) or command line are great choices, VS Code has a great integration too with plenty of extensions.

Then these guides are a good read

* [branching and merging](https://git-scm.com/book/en/v2/Git-Branching-Basic-Branching-and-Merging)
* [GitHub for beginners](https://product.hubspot.com/blog/git-and-github-tutorial-for-beginners)

## [Glossary of ML Terms and Software](https://github.com/albert118/UTS-Professional-Studio-MyRobotPlot/blob/master/Docs/Collecting%20notes.md)

_Click me 🔼_

