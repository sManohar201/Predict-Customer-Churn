# library doc string


# import libraries
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import plot_roc_curve, classification_report

import os
os.environ['QT_QPA_PLATFORM']='offscreen'


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''	
    df = pd.read_csv(pth)
    df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
    return df

def plot_hist(obj, plt_save_pth):
    '''
    Plot histogram for the input and save the plot.
    input:
        obj: pandas series
        plt_name_pth: path to save the plot.
    output:
        None
    '''
    plt.figure(figsize=(20, 10))
    obj.hist();
    plt.savefig(plt_save_pth)
    plt.close()

def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    # TODO: save the path to save images as a constant
    IMAGE_PATH = './images/eda'
    # calculate the churn data
    plot_path = f'{IMAGE_PATH}/churn_distribution.png'
    plot_hist(df['Churn'], plot_path)
    # plot the customer age
    plot_path = f'{IMAGE_PATH}/customer_age_distribution.png'
    plot_hist(df['Customer_Age'], plot_path)
    # plot Matrital status, here I have avoided normalizing the column
    # both gives you similar information.
    plot_path = f'{IMAGE_PATH}/marital_status_distribution.png'
    plot_hist(df['Marital_Status'], plot_path)
    # plot distributions of Total Transaction count
    plt.figure(figsize=(20, 10))
    sns.histplot(df['Total_Trans_Ct'], stat='density', kde=True)
    plt.savefig(f'{IMAGE_PATH}/total_transaction_distribution.png')
    plt.close()
    # plot heatmap of the dataframe
    plt.figure(figsize=(20, 10)) 
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig(f'{IMAGE_PATH}/heatmap.png')
    plt.close()

def encoder_helper(df, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    # make a copy of the dataframe
    old_df = df.copy()
    # gender encoded column
    for ind, cat in enumerate(category_lst):
        lst = []
        groups = old_df.groupby(cat).mean()['Churn']
        lst.extend([groups.loc[val] for val in old_df[cat]])
        # create a new column and add it to the dataframe
        df[response[ind]] = lst    
    return df

def perform_feature_engineering(df, response):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    # TODO: define category columns in constants 
    cat_columns = ['Gender', 'Education_Level',
                   'Marital_Status', 'Income_Category', 'Card_Category']
    response_columns = [f'{cat}_Churn' for cat in cat_columns]
    # new dataframe with the churn data added
    updated_df = encoder_helper(df, cat_columns, response_columns)

    # TODO: define keep_cols in constants
    keep_cols = ['Customer_Age', 'Dependent_count', 'Months_on_book',
             'Total_Relationship_Count', 'Months_Inactive_12_mon',
             'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
             'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
             'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
             'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn', 
             'Income_Category_Churn', 'Card_Category_Churn']
    
    X = pd.DataFrame()
    X[keep_cols] = updated_df[keep_cols]
    X_train, X_test, y_train, y_test = train_test_split(X, updated_df['Churn'], test_size= 0.3, random_state=42)

    return (X_train, X_test, y_train, y_test)


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    # random_forest report
    plt.rc('figure', figsize=(6, 5))
    #plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old approach
    plt.text(0.01, 1.25, str('Random Forest Train'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Random Forest Test'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
    plt.axis('off');
    plt.savefig('./images/results/rf_results.png')
    plt.close();
    # logistic regression report
    plt.rc('figure', figsize=(6, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
    plt.axis('off');
    plt.savefig('./images/results/logistic_results.png')
    plt.close();


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Calculate feature importances
    importances = model.best_estimator_.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20,5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90);
    plt.savefig(f'{output_pth}/feature_importances.png')
    plt.close()

def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    # TODO: create constants file and add random_state, max_iter
    # grid search
    rfc = RandomForestClassifier(random_state=42)
    # logistic regression
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)
    # TODO: add param_grid to constants.py
    param_grid = {
        'n_estimators' : [200, 500],
        'max_features' : ['auto', 'sqrt'],
        'max_depth' : [4, 5, 100],
        'criterion' : ['gini', 'entropy']
    }
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    lrc.fit(X_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)



if __name__ == "__main__":
    path = './data/bank_data.csv'
    df = import_data(path)
    perform_eda(df)