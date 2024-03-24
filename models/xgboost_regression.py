import pickle

# from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
import numpy as np


def train_xgboost_reg(data: pd.DataFrame()):
    label_encoder = LabelEncoder()
    gender_encoder = LabelEncoder()
    married_encoder = LabelEncoder()
    age_bins = np.arange(data['Age'].min(), data['Age'].max() + 3, 3)
    data['Age_Bin'] = pd.cut(data['Age'], bins=age_bins)
    data['Age_Bin_LabelEncoded'] = label_encoder.fit_transform(data['Age_Bin'])
    data['Gender_LabelEncoded'] = gender_encoder.fit_transform(data['Gender'])
    data['Married_LabelEncoded'] = married_encoder.fit_transform(data['Married'])

    with open('pickles/age_bins.pkl', 'wb') as f:
        pickle.dump(age_bins, f)

    with open('pickles/label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)

    with open('pickles/gender_encoder.pkl', 'wb') as f:
        pickle.dump(gender_encoder, f)

    with open('pickles/married_encoder.pkl', 'wb') as f:
        pickle.dump(married_encoder, f)

    data.drop(columns=['Age_Bin', 'Ethnicity', 'Student', 'Rating', 'Balance', 'Gender', 'Married', 'Age'], inplace=True)

    X = data.drop(columns=['Limit'])
    y = data['Limit']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    xgb_reg = XGBRegressor()
    xgb_reg.fit(X_train, y_train)

    # xgb_reg_predictions = xgb_reg.predict(X_test)
    #
    # rmse = mean_squared_error(y_test, xgb_reg_predictions, squared=False)
    # print(f'Combined Model RMSE: {rmse}')
    #
    # mae = mean_absolute_error(y_test, xgb_reg_predictions)
    # print(f'Combined Model MAE: {mae}')

    with open('pickles/xgboost_model.pkl', 'wb') as f:
        pickle.dump(xgb_reg, f)
        print("Finished file")
