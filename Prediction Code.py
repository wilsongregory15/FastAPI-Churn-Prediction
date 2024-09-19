# Wilson Gregory Pribadi - 2602071000
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conint, confloat, constr, Field

model = joblib.load('stacking_model.pkl')
contact_encode= joblib.load('contact_mapping.pkl')
dayofweek_encode= joblib.load('day_of_week_mapping.pkl')
default_encode=joblib.load('default_freq_encoding.pkl')
edu_encode= joblib.load('education_freq_encoding.pkl')
housing_encode=joblib.load('housing_freq_encoding.pkl')
job_encode= joblib.load('job_freq_encoding.pkl')
loan_encode=joblib.load('loan_freq_encoding.pkl')
marital_encode= joblib.load('marital_freq_encoding.pkl')
month_encode=joblib.load('month_mapping.pkl')
poutcome_encode= joblib.load('poutcome_freq_encoding.pkl')
y_encode=joblib.load('response_mapping.pkl')
robust_scaler=joblib.load('robust_scaler.pkl')

app = FastAPI()

class ClientData(BaseModel):
    age: conint(ge=17, le=98)= Field(..., example=30)  
    job: str = Field(..., pattern='^(admin\.|blue\-collar|entrepreneur|housemaid|management|retired|self\-employed|services|student|technician|unemployed|unknown)$', example='student')
    marital: str = Field(..., pattern='^(divorced|married|single|unknown)$', example= 'married')
    education: str = Field(..., pattern='^(basic\.4y|basic\.6y|basic\.9y|high\.school|illiterate|professional\.course|university\.degree|unknown)$', example='basic.4y')
    default: str = Field(..., pattern='^(no|yes|unknown)$', example= 'no')
    housing: str = Field(..., pattern='^(no|yes|unknown)$', example= 'no')
    loan: str = Field(..., pattern='^(no|yes|unknown)$', example= 'no')
    contact: str = Field(..., pattern='^(cellular|telephone)$', example= 'cellular')
    month: str = Field(..., pattern='^(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)$', example= 'may')
    day_of_week: str = Field(..., pattern='^(mon|tue|wed|thu|fri)$', example= 'mon')
    duration: confloat(ge=0, le=3284)= Field(..., example=11.0)  
    campaign: conint(ge=1, le=43)= Field(..., example=3)  
    pdays: conint(ge=0, le=999)= Field(..., example= 3)  
    previous: conint(ge=0, le=6)= Field(..., example=3)  
    poutcome: str = Field(..., pattern='^(failure|nonexistent|success)$', example= 'success')

@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI Model Deployment"}

@app.post("/predict")
def predict(data: ClientData):
    data_dict = data.dict()

    data_dict['job'] = job_encode[data_dict['job']]
    data_dict['marital'] = marital_encode[data_dict['marital']]
    data_dict['education'] = edu_encode[data_dict['education']]
    data_dict['housing'] = housing_encode[data_dict['housing']]
    data_dict['loan'] = loan_encode[data_dict['loan']]
    data_dict['contact'] = contact_encode[data_dict['contact']]
    data_dict['month'] = month_encode[data_dict['month']]
    data_dict['day_of_week'] = dayofweek_encode[data_dict['day_of_week']]
    data_dict['default'] = default_encode[data_dict['default']]
    data_dict['poutcome'] = poutcome_encode[data_dict['poutcome']]

    data_df = pd.DataFrame([data_dict])

    robust_features = ['age', 'duration', 'campaign', 'pdays', 'previous']
    data_df[robust_features] = robust_scaler.transform(data_df[robust_features])

    input_array = np.array(data_df)

    try:
        prediction = model.predict(input_array)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")
    
    result = prediction[0].item() if isinstance(prediction[0], np.generic) else prediction[0]
    return {'prediction': result}