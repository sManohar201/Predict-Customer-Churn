# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
This project involves operationalizing a machine learning model to predict the likelihood of credit card customers churning. For this purpose, a logistic regression model, among others, is trained on the given data using Jupyter Notebook. Our objective is to improve the reusability of the project by modularizing the training pipeline, defining test cases, and implementing logging for the training pipeline.

## Files and data description
Overview of the files and data present in the root directory. 
 - **data**
    * This folder contains the customer data.
 - **images**
    * This folder contains two subfolders 
        - **results**: all the plots related to model analysis.
        - **eda**: data exploration plots.
 - **logs**
    * This folder contains the log files from running the pytest module.
 - **models**
    * This folder contains the trained random forest and logistic regression model.
 - churn_library.py
    * The file contains the implementation of the training pipeline. 
 - test_churn_library.py
    * This file defines the unit test for all the functions implemented in the churn_library.py file.
 - constants.py
    * This defines all the constant variables used in the project.
 - churn_notebook.ipynb
    * This notebook is where the model prototype is defined.
 - pyproject.toml
    * This file contains the configuration for pytest module. See project running section for more details.
 - README.md 
    * Project documenation.
 - requirements_py3.6.txt
    * contains all the relevant packages to install on your local machine.

## Running Files
How do you run your files? What should happen when you run your files?

Pull the repository to your local machine.
- `git clone https://github.com/sManohar201/Predict-Customer-Churn.git`

Create a python environment to run the project. Make sure you have venv installed
- `sudo apt install python3-venv` for Linux/MacOS **or** `pip install virtualenv` for Windows.
- `python3 -m venv <env-name>` for Linux/MacOS **or** `virtualenv <env-name>` for Windows. 
- `<env-name>/bin/activate` for Linux/MacOS **or** `<env-name>/Scripts/Activate.ps1` for Windows.
- `pip install -r <requirements.txt>` to install the dependencies.

Run the training pipeline. `churn_library.py1` is the main file which contains the training pipeline.
- `python churn_library.py`

You can see the images and models folder getting populated. Test the validity of the project by running,
- `pytest` in the project root/parent directory.

This runs unit tests for the functions defined in the `churn_library.py`, which itself is defined in the 
`test_churn_library.py` file. You have couple of options to configure how pytest module should treat the logging messages.

1. You want to write all the log messages into a log file. 
    * This is the default configuration. All the log messages are written into `./logs/churn_library.log`. 
2. You want to print all the log messages on the command prompt.
    * Open `pyproject.toml` file and change `log_cli` to `true`. This should print all the log messages on the command prompt.




