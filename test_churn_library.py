"""
Test module for churn_library.py.

Date: 29/11/2023
"""
import os
import logging
import math
import pytest
import churn_library as cls
import constants as C

os.environ['QT_QPA_PLATFORM'] = 'offscreen'

logging.getLogger(__name__)


def test_import_data():
    '''
    test data import - this example is completed for you to assist with the
                                       other test functions
    '''
    try:
        df = cls.import_data(C.FilePaths.DATA_PATH)
        logging.info("Testing import_data: SUCCESS.")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found.")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows \
							and columns.")
        raise err
    pytest.df = df


def test_perform_eda():
    '''
    test perform eda function
    '''
    data_frame = pytest.df
    try:
        cls.perform_eda(data_frame)
        assert os.path.exists(f"{C.FilePaths.EDA_PATH}/churn_distribution.png")
        assert os.path.exists(
            f"{C.FilePaths.EDA_PATH}/customer_age_distribution.png")
        assert os.path.exists(
            f"{C.FilePaths.EDA_PATH}/marital_status_distribution.png")
        assert os.path.exists(
            f"{C.FilePaths.EDA_PATH}/total_transaction_distribution.png")
        assert os.path.exists(f"{C.FilePaths.EDA_PATH}/heatmap.png")
        logging.info("Testing perform_eda: SUCCESS.")
    except AssertionError as err:
        logging.error("Testing perform_eda: Plots not found!")
        raise err


def test_encoder_helper():
    '''
    test encoder helper
    '''
    data_frame = pytest.df
    try:
        test_df = cls.encoder_helper(data_frame, C.CAT_COLUMNS, C.RES_COLUMNS)
        logging.info("Testing encoder_helper: SUCCESS.")
    except AssertionError as err:
        logging.error("Testing encoder_helper: Function failed to run!")
        raise err

    try:
        for cat in C.RES_COLUMNS:
            assert cat in test_df.columns
        logging.info("Testing encoder_helper: SUCCESS - All categories found.")
    except AssertionError as err:
        logging.error("Testing encoder_helper: Categories are missing!")
        raise err


def test_perform_feature_engineering():
    '''
    test perform_feature_engineering
    '''
    data_frame = pytest.df
    try:
        X_train, X_test, y_train, y_test = cls.perform_feature_engineering(
            data_frame, C.RES_COLUMNS)
        assert X_train.shape[0] > 0
        assert X_train.shape[1] > 0
        assert X_test.shape[0] > 0
        assert X_test.shape[1] > 0
        assert y_train.shape[0] > 0
        assert y_test.shape[0] > 0
        assert data_frame.shape[0] == (X_test.shape[0] + X_train.shape[0])
        assert data_frame.shape[0] == (y_train.shape[0] + y_test.shape[0])
        assert X_train.shape[1] == len(C.KEEP_COLS)
        assert math.ceil(data_frame.shape[0] * 0.3) == X_test.shape[0]
        logging.info("Testing perform_feature_engineering: SUCCESS.")
    except AssertionError as err:
        logging.error(
            "Testing perform_feature_engineering: The train/test has \
							failed.")
        raise err
    pytest.x_train = X_train
    pytest.x_test = X_test
    pytest.y_train = y_train
    pytest.y_test = y_test


def test_train_models():
    '''
    test train_models
    '''
    try:
        cls.train_models(pytest.x_train, pytest.x_test,
                         pytest.y_train, pytest.y_test)
        logging.info("Testing train_models: SUCCESS - Training finished.")
    except Exception as err:
        logging.error("Testing train_model: Training failed!")
        raise err
    try:
        assert os.path.exists(f"{C.FilePaths.RESULTS_PATH}/rf_results.png")
        assert os.path.exists(
            f"{C.FilePaths.RESULTS_PATH}/logistic_results.png")
        assert os.path.exists(
            f"{C.FilePaths.RESULTS_PATH}/feature_importances.png")
        assert os.path.exists(
            f"{C.FilePaths.RESULTS_PATH}/roc_curve_results.png")
        logging.info(
            "Testing train_models: SUCCESS - Results plotting finished.")
    except AssertionError as err:
        logging.error("Testing train_models: Results plotting failed!")
        raise err
    try:
        assert os.path.exists(f"{C.FilePaths.MODELS_PATH}/rfc_model.pkl")
        assert os.path.exists(f"{C.FilePaths.MODELS_PATH}/logistic_model.pkl")
        logging.info("Testing train_models: SUCCESS - Models saved.")
    except AssertionError as err:
        logging.error("Testing train_models: Models are not saved.")
        raise err
