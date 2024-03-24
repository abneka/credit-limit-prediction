import os

from fastapi import FastAPI

from models.linear_regression import train_linear_reg
from models.xgboost_regression import train_xgboost_reg
from routes.connection import router as connection
from routes.predict_xgboost import router as predict_xgboost_regression
from routes.predict_lin_reg import router as predict_linear_regression
import nest_asyncio
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

credit_data = pd.read_csv("credit.csv")
credit_data = credit_data[credit_data['Age'] <= 80]
credit_data = credit_data[credit_data.Cards < 6]
credit_data['Income'] = credit_data['Income']*1000

if not os.path.exists('pickles/lin_reg_model.pkl'):
    train_linear_reg(credit_data)
# if not os.path.exists('pickles/xgboost_model.pkl'):
#     train_xgboost_reg(credit_data)

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# app.include_router(login)
app.include_router(connection)
app.include_router(predict_linear_regression)
app.include_router(predict_xgboost_regression)

if __name__ == "__main__":
    nest_asyncio.apply()
    uvicorn.run(app, host="localhost", port=8000)

