"""
Module contains constants used by churn_library.py
Author: Sabari 
Date: 29/11/2023
"""

# FilePath object holds all the folderpaths.
class FilePaths:
    """
    Holds all the parent file paths.
    """
    DATA_PATH = './data/bank_data.csv'
    RESULTS_PATH = './images/results'
    EDA_PATH = './images/eda'
    MODELS_PATH = './models'


# category columns list
CAT_COLUMNS = ['Gender', 'Education_Level',
               'Marital_Status', 'Income_Category', 'Card_Category']

# response columns list
RES_COLUMNS = [f'{cat}_Churn' for cat in CAT_COLUMNS]

# updated column list
KEEP_COLS = ['Customer_Age', 'Dependent_count', 'Months_on_book',
             'Total_Relationship_Count', 'Months_Inactive_12_mon',
             'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
             'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
             'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
             'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
             'Income_Category_Churn', 'Card_Category_Churn']

# object to hold all the hyperparameters
class Hyperparameter:
    """
    Holds all the parameters used by the random forest and linear regression
    models.
    """
    RANDOM_STATE = 42
    TEST_SIZE = 0.3
    SOLVER = 'lbfgs'
    MAX_ITER = 3000
    PARAM_GRID = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }
    CROSS_VAL = 5
