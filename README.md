# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
This is a project to create model for churn predictions.
Code follows PEP8 code style.

## Files and data description
churn_library.py - set of functions for churn prediction.

churn_script_logging_and_tests.py - tests for functions in churn_library.py are executed by running this script. 

data/bank_data.csv - original dataset.

images/eda - images generated for exploratory data analysis.

images/results - images generated based on results of modelling.

logs/churn_library.log - log of the tests.

models/logisitc_model.pkl - saved logistic regression model.

models/rfc_model.pkl - saved random forest model.


## Running Files

To execute chunk_library.py, run "python.exe chunk_library.py" in terminal. This reads data, creates plots in the images/eda, builds model,
saves model into models folder and results into images/results.

To execute churn_script_logging_and_tests.py, run "python.exe churn_script_logging_and_tests.py" in terminal. 
Results of the tests are logged in ./logs/churn_library.log.



