"""
Library of tests for functions in churn_library.py.

Author:Maxim Shatskiy
Date: 25.08.2022
"""
import os
import logging
import churn_library as cls

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_add_response_variable(add_response_variable):
    '''
    test perform eda function
    '''

    df = cls.import_data("./data/bank_data.csv")
    try:
        response = 'Churn'
        df_new = add_response_variable(df.copy(), response)
        logging.info("Testing add_response_variable: SUCCESS")
    except Exception as exception:
        logging.error("Testing add_response_variable: FAILURE. Unknown error.")
        raise exception

    try:
        assert df_new.shape[0] > 0
        assert df_new.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing add_response_variable: The returned data frame doesn't appear to have rows and columns")
        raise err

    try:
        assert df_new.shape[1] > df.shape[1]
    except AssertionError as err:
        logging.error(
            "Testing add_response_variable: "
            "the resulting dataframe has not more columns than original one")
        raise err

    try:
        assert response in df_new.columns
    except AssertionError as err:
        logging.error(
            "Testing add_response_variable: "
            "response variable is not in the columns of the new dataframe")
        raise err


def test_eda(perform_eda):
    '''
    test perform_eda function
    '''

    df = cls.import_data("./data/bank_data.csv")
    response = 'Churn'
    df = cls.add_response_variable(df, response)
    try:
        perform_eda(df)
        logging.info("Testing perform_eda: SUCCESS")
    except FileNotFoundError:
        logging.info(
            "Testing perform_eda: EDA could not be performed. File was not found.")

    try:
        assert os.path.exists('./images/eda/churn_distribution.png')
        assert os.path.exists('./images/eda/customer_age_distribution.png')
        assert os.path.exists('./images/eda/martial_status_distribution.png')
        assert os.path.exists(
            './images/eda/total_transaction_distirbution.png')
        assert os.path.exists('./images/eda/heatmap.png')
        logging.info(
            "Testing perform_eda: perform_eda SUCCESS. All required files were generated.")
    except AssertionError as err:
        logging.error(
            "Testing perform_eda: one of the files was not generated")
        raise err


def test_encoder_helper(encoder_helper):
    '''
    test encoder helper
    '''
    df = cls.import_data("./data/bank_data.csv")
    response = 'Churn'
    df = cls.add_response_variable(df, response)
    try:
        cat_columns = [
            'Gender',
            'Education_Level',
            'Marital_Status',
            'Income_Category',
            'Card_Category'
        ]
        df_new = encoder_helper(df.copy(), cat_columns, response)
        logging.info("Testing encoder_helper: SUCCESS")
    except Exception as exception:
        logging.error("Testing encoder_helper: FAILURE. Unknown error.")
        raise exception

    try:
        assert df_new.shape[0] > 0
        assert df_new.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: The returned data frame doesn't appear to have rows and columns")
        raise err

    try:
        assert df_new.shape[1] == df.shape[1] + len(cat_columns)
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: "
            "number of columns in a new dataframe is not equal to expected number of columns")
        raise err


def test_perform_feature_engineering(perform_feature_engineering):
    '''
    test perform_feature_engineering
    '''
    df = cls.import_data("./data/bank_data.csv")
    response = 'Churn'
    df = cls.add_response_variable(df, response)

    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']
    try:
        X_train, X_test, y_train, y_test = perform_feature_engineering(df)
        logging.info("Testing perform_feature_engineering: SUCCESS")
    except Exception as exception:
        logging.error("Testing add_response_variable: FAILURE. Unknown error.")
        raise exception

    try:
        assert X_train.shape[0] > 0
        assert X_train.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing perform_feature_engineering: The returned X_train doesn't appear to have rows and columns")
        raise err

    try:
        assert X_test.shape[0] > 0
        assert X_test.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing perform_feature_engineering: The returned X_test doesn't appear to have rows and columns")
        raise err

    try:
        assert all(item in X_train.columns for item in keep_cols)
    except AssertionError as err:
        logging.error(
            "Testing perform_feature_engineering: "
            "not all columns that should be kept are in the columns of the dataframe")
        raise err

    try:
        assert X_train.shape[0] == y_train.shape[0]
    except AssertionError as err:
        logging.error(
            "Testing perform_feature_engineering: "
            "features and response variables in the training set have different number of rows")
        raise err


def test_train_models(train_models):
    '''
    test train_models
    '''

    df = cls.import_data("./data/bank_data.csv")
    response = 'Churn'
    df = cls.add_response_variable(df, response)
    X_train, X_test, y_train, y_test = cls.perform_feature_engineering(df)

    try:
        train_models(X_train, X_test, y_train, y_test)
        logging.info("Testing train_models: SUCCESS")
    except Exception as exception:
        logging.error("Testing train_models: FAILURE. Unknown error.")
        raise exception

    try:
        assert os.path.exists('./models/logistic_model.pkl')
        assert os.path.exists('./models/rfc_model.pkl')
        assert os.path.exists('./images/results/roc_curve_results.png')
        assert os.path.exists('./images/results/feature_importances.png')
        assert os.path.exists('./images/results/rf_results.png')
        assert os.path.exists('./images/results/logistics_results.png')
        logging.info(
            "Testing train_models: perform_eda SUCCESS. All required files were generated.")
    except AssertionError as err:
        logging.error(
            "Testing train_models: one of the files was not generated")
        raise err


if __name__ == "__main__":
    test_import(cls.import_data)
    test_add_response_variable(cls.add_response_variable)
    test_eda(cls.perform_eda)
    test_encoder_helper(cls.encoder_helper)
    test_perform_feature_engineering(cls.perform_feature_engineering)
    test_train_models(cls.train_models)
