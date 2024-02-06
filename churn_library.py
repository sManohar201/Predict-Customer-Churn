"""
chrun_library.py module contains functions of churn customer analysis.
Author: Sabari 
Date: 29/11/2023
"""

# import libraries
import os
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import constants as C
sns.set()


os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    df = pd.read_csv(pth)
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
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
    obj.hist()
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
    # calculate the churn data
    plot_path = f'{C.FilePaths.EDA_PATH}/churn_distribution.png'
    plot_hist(df['Churn'], plot_path)
    # plot the customer age
    plot_path = f'{C.FilePaths.EDA_PATH}/customer_age_distribution.png'
    plot_hist(df['Customer_Age'], plot_path)
    # plot Matrital status, here I have avoided normalizing the column
    # both gives you similar information.
    plot_path = f'{C.FilePaths.EDA_PATH}/marital_status_distribution.png'
    plot_hist(df['Marital_Status'], plot_path)
    # plot distributions of Total Transaction count
    plt.figure(figsize=(20, 10))
    sns.histplot(df['Total_Trans_Ct'], stat='density', kde=True)
    plt.savefig(f'{C.FilePaths.EDA_PATH}/total_transaction_distribution.png')
    plt.close()
    # plot heatmap of the dataframe
    plt.figure(figsize=(20, 10))
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig(f'{C.FilePaths.EDA_PATH}/heatmap.png')
    plt.close()


def encoder_helper(df, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for
            naming variables or index y column]

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
              response: string of response name [optional argument that could be used for
              naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    # new dataframe with the churn data added
    updated_df = encoder_helper(df, C.CAT_COLUMNS, response)
    X = pd.DataFrame()
    X[C.KEEP_COLS] = updated_df[C.KEEP_COLS]
    X_train, X_test, y_train, y_test = train_test_split(X, updated_df['Churn'],
                                                        test_size=C.Hyperparameter.TEST_SIZE,
                                                        random_state=C.Hyperparameter.RANDOM_STATE)
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
    # plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old approach
    plt.text(0.01, 1.25, str('Random Forest Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Random Forest Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig(f'{C.FilePaths.RESULTS_PATH}/rf_results.png')
    plt.close()
    # logistic regression report
    plt.rc('figure', figsize=(6, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig(f'{C.FilePaths.RESULTS_PATH}/logistic_results.png')
    plt.close()


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
    plt.figure(figsize=(20, 5))
    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')
    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])
    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)
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
    # grid search
    rfc = RandomForestClassifier(random_state=C.Hyperparameter.RANDOM_STATE)
    # logistic regression
    lrc = LogisticRegression(solver=C.Hyperparameter.SOLVER,
                             max_iter=C.Hyperparameter.MAX_ITER)
    cv_rfc = GridSearchCV(estimator=rfc,
                          param_grid=C.Hyperparameter.PARAM_GRID,
                          cv=C.Hyperparameter.CROSS_VAL)
    cv_rfc.fit(X_train, y_train)
    # train logistic regression model
    lrc.fit(X_train, y_train)
    # predictions for random forest
    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)
    # predictions for logistic regression
    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)
    # save models to local directory
    joblib.dump(
        cv_rfc.best_estimator_,
        f'{C.FilePaths.MODELS_PATH}/rfc_model.pkl')
    joblib.dump(lrc, f'{C.FilePaths.MODELS_PATH}/logistic_model.pkl')
    # plot classification report images
    classification_report_image(
        y_train,
        y_test,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf)
    feature_importance_plot(cv_rfc, X_train, C.FilePaths.RESULTS_PATH)
    lrc_plot = plot_roc_curve(lrc, X_test, y_test)
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    rfc_disp = plot_roc_curve(cv_rfc, X_test, y_test, ax=ax, alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)
    plt.savefig(f'{C.FilePaths.RESULTS_PATH}/roc_curve_results.png')
    plt.close()


def run_pipeline():
    """
    This functions creates a pipeline to all the necessary functions
    in respective order.
    """
    df = import_data(C.FilePaths.DATA_PATH)
    perform_eda(df)
    X_train, X_test, y_train, y_test = perform_feature_engineering(
        df, C.RES_COLUMNS)
    train_models(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    run_pipeline()
