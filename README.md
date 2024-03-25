# Credit limit Prediction

Dataset used: [Credit Card Limit Prediction](https://www.kaggle.com/datasets/syedasimalishah/credit-card-limit-prediction/data)

## Observations from initial data analysis
1. Student: 85% of the data is marked as not a student while out of the remaining 15%, the ages are in between 40 and 80. So it is not practical to use this data as 40+ are generally not students.
2. Ethnicity: Dropping Ethnicity as the data contains just 3 ethnicities and no clear pattern between them either.
3. Income: There is a clear correlation between Income and limit.
4. Age: There is a recurring pattern between Age and Limit.
5. Cards: There is a slight pattern between cards and Limit which might be more clear if there is more data available.
6. Education: Similar pattern to age bins.
7. Married: Married people have slightly higher limit than unmarried.
8. Gender: Females have slightly higher limit.

## API endpoints on Render
1. You can view and manage your endpoints on [Render Dashboard](https://dashboard.render.com/) (Log in with GitHub).
2. The endpoint for this repo is [https://credit-limit-prediction.onrender.com](https://credit-limit-prediction.onrender.com), and you can view available APIs with [Swagger UI](https://credit-limit-prediction.onrender.com/docs#).
3. Import the following `curl` commands to check the predictions:
   - For linear regression:
     ```bash
     curl --location --request GET 'https://credit-limit-prediction.onrender.com/predict/linear_regression' \
     --header 'accept: application/json' \
     --header 'income: 20000' \
     --header 'Content-Type: application/json' \
     --data '{
         "income": "20000"
     }'
     ```
   - For XGBoost predictions:
     ```bash
     curl --location --request GET 'https://credit-limit-prediction.onrender.com/predict/xgboost_regression' \
     --header 'accept: application/json' \
     --header 'Content-Type: application/json' \
     --data '{
         "Income": 155000,
         "Age": 40,
         "Gender": "Female",
         "Married": "No",
         "Cards": 3,
         "Education": 11
     }'
     ```
4. The free tier of render spins down on inactivity. So if you feel the API is slow, give it a couple of minutes for the first 2 calls. It will work fast after that. (Or setup a cron job on some free website)
