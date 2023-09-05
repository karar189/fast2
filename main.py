from fastapi import FastAPI
import numpy as np
import pickle
from cropData import CropData


app = FastAPI()
pickle_in = open("NBClassifier.pkl", "rb")
classifier = pickle.load(pickle_in)


@app.get("/")
async def hello():
    return "Hello Crops"

#create a post

@app.post("/predict")
async def predict_crop(data: CropData):
    data = data.dict()
    n = data["n"]
    p = data["p"]
    k = data["k"]
    temperature = data["temperature"]
    humidity = data["humidity"]
    ph = data["ph"]
    rainfall = data["rainfall"]
    prediction = classifier.predict([[n, p, k, temperature, humidity, ph, rainfall]])
    return {"crop": prediction[0]}


