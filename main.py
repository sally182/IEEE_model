# Define the FastAPI application
from typing import List, Union
from typing_extensions import Annotated
from fastapi import FastAPI, Query
import joblib
import numpy as np




app = FastAPI()


@app.get("/")
def read_root():
    def scale_and_predict(data):
        
        scaler = joblib.load("scaler.gz")
        model = joblib.load("model.gz")
        data = np.array(data)
        print(data)
        scaled_data = scaler.transform(data)
        pred = model.predict([scaled_data])
    return pred

    
    
    return {pred}


@app.post("/predict")
async def predict(q: Annotated[Union[List[float], None], Query()] = None):
    data=q
    #prediction = scale_and_predict(data)
    return {"prediction": data}
