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
    churn_data = df['Attrition_Flag'].apply(
                lambda val: 0 if val == "Existing Customer" else 1)
    churn_data_plot_path = f'{IMAGE_PATH}/churn_distribution.png'
    plot_hist(churn_data, churn_data_plot_path)
    # plot the customer age
    customer_age_plot_path = f'{IMAGE_PATH}/customer_age_distribution.png'
    plot_hist(df['Customer_Age'], customer_age_plot_path)
    # plot Matrital status, here I have avoided normalizing the column
    # both gives you similar information.
    marital_status_plot_path = f'{IMAGE_PATH}/marital_status_distribution.png'
    plot_hist(df['Marital_Status'], marital_status_plot_path)
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
    pass


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
    pass


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
    pass

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
    pass

if __name__ == "__main__":
    path = './data/bank_data.csv'
    df = import_data(path)
    perform_eda(df)