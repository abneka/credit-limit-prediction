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

    if income < 20000 or income > 200000:
        return JSONResponse(content="Enter value between 20000 and 200000", status_code=200)
    else:
        with open('pickles/lin_reg_model.pkl', 'rb') as f:
            loaded_model = pickle.load(f)
        input_income = [[income]]
        prediction = loaded_model.predict(input_income)

        return JSONResponse(content={"prediction": prediction.tolist()}, status_code=200)





