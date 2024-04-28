import pickle

from fastapi import APIRouter, Request, HTTPException
from starlette.responses import JSONResponse
import pandas as pd

router = APIRouter()


@router.post("/predict/xgboost_regression") # change GET method to POST
async def pred_xr(request: Request):
    data = await request.json()
    if request is None:
        raise HTTPException(status_code=500, detail="No request json")

    income = data.get('Income')
    age = data.get('Age')
    married = data.get('Married')
    cards = data.get('Cards')
    education = data.get('Education')
    gender = data.get('Gender')

    if income is None:
        return JSONResponse(content="Income is missing", status_code=400)
    if age is None:
        return JSONResponse(content="Age is missing", status_code=400)
    if gender is None:
        return JSONResponse(content="Gender is missing", status_code=400)
    if education is None:
        return JSONResponse(content="Education is missing", status_code=400)
    if cards is None:
        return JSONResponse(content="Cards is missing", status_code=400)
    if married is None:
        return JSONResponse(content="Married is missing", status_code=400)

    if income < 1000000 or income > 25000000:
        return JSONResponse(content="Enter Income value between 1000000 UZS and 25000000 UZS", status_code=200)
    if age < 20 or age > 80:
        return JSONResponse(content="Enter Age value between 20 and 80", status_code=200)

    df = pd.DataFrame([data])
    df['Income'] = int((income * 12) / 12500)

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
    output = prediction.tolist()[0]*12500
    response_content = {f"prediction": f"{output} UZS"}
    return JSONResponse(content=response_content, status_code=200)




