from datetime import datetime
import time

from fastapi import FastAPI
from pydantic import BaseModel
from loguru import logger

from classifier import NewsCategoryClassifier


class PredictRequest(BaseModel):
    source: str
    url: str
    title: str
    description: str


class PredictResponse(BaseModel):
    scores: dict
    label: str


MODEL_PATH = "../data/news_classifier.joblib"
LOGS_OUTPUT_PATH = "../data/logs.out"

app = FastAPI()


@app.on_event("startup")
def startup_event():
    global clf
    clf = NewsCategoryClassifier()
    clf.load(MODEL_PATH)
    logger.add(LOGS_OUTPUT_PATH)
    logger.info("Setup completed")


@app.on_event("shutdown")
def shutdown_event():
    # clean up
    logger.remove()
    logger.info("Shutting down application")


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    # get model prediction for the input request
    # construct the data to be logged
    # construct response
    start_time = time.time()
    probabilities = clf.predict_proba(request.description)
    label = clf.predict_label(request.description)
    end_time = time.time()
    latency = int((end_time - start_time) * 1000)  # in milliseconds

    log_data = {
        "timestamp": datetime.now().strftime("%Y:%m:%d %H:%M:%S"),
        "request": request,
        "prediction": {'scores': probabilities, 'label': label},
        "latency": latency
    }
    logger.info(log_data)
    
    response = PredictResponse(scores=probabilities, label=label)
    return response


@app.get("/")
def read_root():
    return {"Hello": "World"}
