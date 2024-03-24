import pickle

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error


def train_linear_reg(data: pd.DataFrame()):
    print(data.head(3))
    X = data.drop(columns=['Limit'])
    y = data['Limit']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    lin_reg = LinearRegression()
    lin_reg.fit(X_train[['Income']], y_train)

    # lin_reg_predictions = lin_reg.predict(X_test[['Income']])

    # rmse = mean_squared_error(y_test, lin_reg_predictions, squared=False)
    # print(f'Linear Model RMSE: {rmse}')
    #
    # mae = mean_absolute_error(y_test, lin_reg_predictions)
    # print(f'Linear Model MAE: {mae}')

    with open('pickles/lin_reg_model.pkl', 'wb') as f:
        pickle.dump(lin_reg, f)
        print("Finished file")