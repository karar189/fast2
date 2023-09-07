from fastapi import FastAPI
import numpy as np
import pickle
from fastapi.middleware.cors import CORSMiddleware
from cropData import CropData


app = FastAPI()
pickle_in = open("NBClassifier.pkl", "rb")
classifier = pickle.load(pickle_in)

# Define a list of allowed origins (replace '*' with specific origins in production)
allowed_origins = ["*"]

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,  # Allow sending cookies
    allow_methods=["*"],     # Allow all HTTP methods
    allow_headers=["*"],     # Allow all HTTP headers
)


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


