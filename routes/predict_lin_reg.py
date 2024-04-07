import pickle

from fastapi import APIRouter, Request, HTTPException
from starlette.responses import JSONResponse

router = APIRouter()


@router.get("/predict/linear_regression")
async def pred_lr(request: Request):
    data = await request.json()
    if request is None:
        raise HTTPException(status_code=500, detail="No request json")

    income = data.get('income')

    if income is None:
        return JSONResponse(content="Parameter 'income' is missing", status_code=400)

    try:
        income = int(income)
    except ValueError:
        return JSONResponse(content="Invalid value for 'income'", status_code=400)

    if income < 1000000 or income > 25000000:
        return JSONResponse(content="Enter Income value between 1000000 UZS and 25000000 UZS", status_code=200)
    else:
        converted_income = int((income * 12) / 12500)
        with open('pickles/lin_reg_model.pkl', 'rb') as f:
            loaded_model = pickle.load(f)
        input_income = [[converted_income]]
        prediction = loaded_model.predict(input_income)

        return JSONResponse(content={"prediction": prediction.tolist()}, status_code=200)





