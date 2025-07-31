"""
Iris Classifier API

A FastAPI application that serves predictions for the Iris flower classification model.
"""

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

MODEL = joblib.load("model.joblib")


class IrisIn(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


class IrisOut(BaseModel):
    class_id: int
    class_name: str


app = FastAPI(title="Iris Classifier API")


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict", response_model=IrisOut)
def predict(inp: IrisIn):
    data = np.array(
        [[inp.sepal_length, inp.sepal_width, inp.petal_length, inp.petal_width]]
    )
    pred = MODEL.predict(data)[0]
    names = ["setosa", "versicolor", "virginica"]
    return {"class_id": int(pred), "class_name": names[pred]}
