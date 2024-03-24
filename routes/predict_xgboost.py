import pickle

from fastapi import APIRouter, Request, HTTPException
from starlette.responses import JSONResponse
import pandas as pd

router = APIRouter()


@router.get("/predict/xgboost_regression")
async def pred_xr(request: Request):
    data = await request.json()

    if request is None:
        raise HTTPException(status_code=500, detail="No request json")
    df = pd.DataFrame([data])

    with open('pickles/xgboost_model.pkl', 'rb') as f:
        loaded_model = pickle.load(f)

    with open('pickles/age_bins.pkl', 'rb') as f:
        age_bins = pickle.load(f)

    with open('pickles/label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)

    with open('pickles/gender_encoder.pkl', 'rb') as f:
        gender_encoder = pickle.load(f)

    with open('pickles/married_encoder.pkl', 'rb') as f:
        married_encoder = pickle.load(f)

    df['Age_Bin'] = pd.cut(df['Age'], bins=age_bins)
    df['Age_Bin_LabelEncoded'] = label_encoder.transform(df['Age_Bin'])
    df['Gender_LabelEncoded'] = gender_encoder.transform(df['Gender'])
    df['Married_LabelEncoded'] = married_encoder.transform(df['Married'])

    # Drop unnecessary columns
    df.drop(columns=['Age', 'Gender', 'Married', 'Age_Bin'], inplace=True)
    prediction = loaded_model.predict(df)

    return JSONResponse(content={"prediction": prediction.tolist()}, status_code=200)




