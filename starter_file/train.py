import argparse
import logging
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
import pandas as pd
import json
from azureml.core.run import Run
from azureml.core import Workspace, Dataset, Datastore
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error

def get_data():
    '''
    Reads raw dataset uploaded from Kaggle and stored in the datastore.
    '''
    
    df = pd.read_csv("https://storage.googleapis.com/kagglesdsdata/datasets/3186183/5526698/Walmart%20Data%20Analysis%20and%20Forcasting.csv?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20230801%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20230801T183427Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=5be97b22bea1d8e7561b7fd1ce98a17e7d5edddd533590bb3fe8ddb0928d70b43f611c4dc0a9c0ce3c1eec2589169823e5ac13c57635e02b9c9a6add38ba20779c28bc1f7d8c0ab987b95f5cc59b0f7bffc66aa0067861ace20a163d65049a3acd27aeef4c0014c1b57488d601654dd283b5ee4e91f348b21360732716f693499313ab7fe40066440726fcf0e94a084d93051e44a10e93aeb531b160f185c4a980b55b815105438f703949f02f2193201c0dc05491736a9861010b3a1625852539d77da9dcfb2f08d222e9414e98a9380d1267ad8ff675352a68f50c62fedcf0fdba09ecf02aefa868fe75a1195eee9102eab24d418dfd6f2a815b1470a2e9bc")
    return df

def process_data(df):
    '''
    This function formats the dataframe, adding past 8 weeks of sales as lagged features
    and 4 weeks of future sales as the label column.

    Return:
    x_train. Training data with features + lagged sales
    x_val. Validation data with features + lagged sales
    y_train. training dependent vector with the next 4 weeks of sales
    y_val. training dependent vector with the next 4 weeks of sales
    '''
    
    df_with_windows = []
    for store_num in df.Store.unique():
        store_df = df[df.Store == store_num].copy()
        # making lag features
        for i in range(1, 9):
            store_df[f'Weekly_Sales_t-{i}'] = store_df['Weekly_Sales'].shift(i)
        # making future_time_steps
        for i in range(1,4):
            store_df[f'Weekly_Sales_t+{i}'] = store_df['Weekly_Sales'].shift(-i)

        df_with_windows.append(store_df)

    df_with_windows = pd.concat(df_with_windows).dropna()
    # renaming first future value, to follow the same pattern as the other columns
    df_with_windows.rename(columns={"Weekly_Sales":"Weekly_Sales_t+0"}, inplace=True)
    df_with_windows = df_with_windows[['Store', 'Date', 'Holiday_Flag', 'Temperature',
                                        'Fuel_Price', 'CPI', 'Unemployment', 'Weekly_Sales_t-1',
                                        'Weekly_Sales_t-2', 'Weekly_Sales_t-3', 'Weekly_Sales_t-4',
                                        'Weekly_Sales_t-5', 'Weekly_Sales_t-6', 'Weekly_Sales_t-7',
                                        'Weekly_Sales_t-8', 'Weekly_Sales_t+0','Weekly_Sales_t+1', 'Weekly_Sales_t+2',
                                        'Weekly_Sales_t+3']]

    # separate by store, train_test_split, and then join data again
    x_train, x_val, y_train, y_val = [], [], [], []

    for store_num in df_with_windows.Store.unique():
        store_df = df_with_windows[df_with_windows.Store == store_num].copy()
        # future columns filter
        ftr = store_df.columns.str.match(r'.+t\+\d')
        # making label vector
        y_store = store_df.loc[:, ftr].apply(lambda row: list(row), axis=1).tolist()
        # convert list to numpy array format
        y_store = np.array(y_store)
        # making training data
        X_store = store_df.drop(columns='Date').values
        x_train_store, x_val_store, y_train_store, y_val_store = train_test_split(X_store, y_store, test_size=0.2, shuffle=False, random_state=96)
        
        # appending to final results
        x_train.append(x_train_store)
        x_val.append(x_val_store)
        y_train.append(y_train_store)
        y_val.append(y_val_store)

    x_train = np.concatenate(x_train)
    x_val = np.concatenate(x_val)
    y_train = np.concatenate(y_train)
    y_val = np.concatenate(y_val)

    return x_train, x_val, y_train, y_val

def train_model(x_train, y_train, kwargs):
    '''
    This function receives the processed X, y values and fits a multiple output XGBRegressor to it.

    Returns: the fitted model
    '''
    #Define the estimator
    estimator = XGBRegressor(
        objective = 'reg:squarederror',
        **kwargs
        )

    # Define the model
    my_model = MultiOutputRegressor(estimator = estimator, n_jobs = -1)
    my_model.fit(x_train, y_train)

    return my_model

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--learning_rate', type=float, default=0.1, help="The maximum depth per tree")
    parser.add_argument('--max_depth', type=int, default=5, help="The maximum depth per tree")
    parser.add_argument('--n_estimators', type=int, default=100, help="The number of trees in the ensemble")
    parser.add_argument('--reg_lambda', type=float, default=1, help="L2 regularization on the weights")
    parser.add_argument('--subsample', type=float, default=1, help="fraction of observations to be sampled for each tree")
    parser.add_argument('--colsample_bytree', type=float, default=1, help="fraction of columns to be randomly sampled for each tree.")

    args = parser.parse_args()

    run = Run.get_context()

    run.log("Learning rate", float(args.learning_rate))
    run.log("Max depth", int(args.max_depth))
    run.log("Number of estimators", int(args.n_estimators))
    run.log("L2 regularization Strength" , float(args.reg_lambda))
    run.log("Subsample", float(args.subsample))
    run.log("Fraction of cols sampled by tree", float(args.colsample_bytree))

    # building hyperparameters dictionary to be passed to the model
    params_dict = {
    'learning_rate': float(args.learning_rate),
    'max_depth': int(args.max_depth),
    'n_estimators': int(args.n_estimators),
    'lambda' : float(args.reg_lambda),
    'subsample' : float(args.subsample),
    'colsample_bytree': float(args.colsample_bytree)
    }

    logging.basicConfig(
    level=logging.INFO,
    format="%(name)s - %(asctime)s - %(levelname)s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
    )

    logging.info('Getting data...')
    df = get_data()
    # computing values to normalized RMSE just like in the automl experiment
    y_min, y_max = df['Weekly_Sales'].min(), df['Weekly_Sales'].max()
    
    logging.info('formatting data...')
    x_train, x_val, y_train, y_val = process_data(df)

    logging.info('Training models...')
    model = train_model(x_train, y_train, params_dict)
    logging.info("Model object: ")
    logging.info(model)
    y_pred = model.predict(x_val)

    # computing Normalized RMSE
    nrmse = np.sqrt(mean_squared_error(y_val, y_pred))/(y_max - y_min)

    # save model as run output
    os.makedirs('outputs',exist_ok = True)
    joblib.dump(model,'outputs/model.joblib')

    run.log("NRMSE", float(nrmse))

    
    

if __name__ == '__main__':
    main()