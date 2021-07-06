from flask import Flask, request
import json
import pandas as pd
import joblib    

app = Flask(__name__)
forest = joblib.load("forest_v1.joblib")

@app.route("/")#Provide route
def index():#Function tells what to do when you arrive at that route
    return "Hello World" #200 is All Good

#This is our API endpoint
@app.route("/predict", methods=['GET'])#This endpoint is a GET endpoint, can deliver information to it
def predict():
    # Take API payload as a JSON format
    json_ = request.json
    df = pd.read_json(json_)
    prediction = forest.predict(df)
    #Cant return an ndarray, so turn list and pack to dictionary
    return {"prediction": list(prediction)}
#Returns error as its expecting a payload, due to the request function

app.run()#Can add port = 5001 for different

