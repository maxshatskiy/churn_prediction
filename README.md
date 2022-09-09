# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
This is a project to create a model for prediction of credit card customers that are most likely to churn.
Code follows PEP8 code style and follows engineering best practices for implementing software: modular, documented,
and tested.

The data used are taken from: https://www.kaggle.com/sakshigoyal7/credit-card-customers/code

## Files and data description
- **churn_library.py** - set of functions for churn prediction.

- **churn_script_logging_and_tests.py** - tests for functions in churn_library.py are executed by running this script. 

- **data/bank_data.csv** - original dataset.

- **images/eda** - images generated for exploratory data analysis.

- **images/results** - images generated based on results of modelling.

- **logs/churn_library.log** - log of the tests.

- **models/logisitc_model.pkl** - saved logistic regression model.

- **models/rfc_model.pkl** - saved random forest model.

- **requirements_py3.8.txt** - contains list of packages required to run this package. The required packages are also 
listed below:
  - scikit-learn==0.24.1
  - shap==0.40.0
  - joblib==1.0.1
  - pandas==1.2.4
  - numpy==1.20.1
  - matplotlib==3.3.4
  - seaborn==0.11.2
  - pylint==2.7.4
  - autopep8==1.5.6
  - pytest==7.1.2
  
## Running Files
In order to install required packages using **requirements_py3.8.txt** run the following command:
"*python -m pip install -r requirements_py3.8.txt*".

The package can be run interactively or from the command-line interface.

To execute chunk_library.py, run "*python.exe chunk_library.py*" in terminal. This reads data, creates plots in the images/eda, builds model,
saves model into models folder and results into images/results.

To execute churn_script_logging_and_tests.py, run "*python.exe churn_script_logging_and_tests.py*" in terminal. 
Results of the tests are logged in ./logs/churn_library.log.



