import pickle 
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd


app=Flask(__name__)#Basic flask is created here
#__name__ will the strting point of application from whre it will start
##Load the model
regmodel=pickle.load(open("regmodel.pkl","rb"))
scalar=pickle.load(open("scaling.pkl","rb"))
@app.route("/")
def home():
    return render_template("home.html")# it will basically be a html page 
#by default if i hit this flask app it will redirect to the home.html

@app.route("/predict_api",methods=["POST"])
#using post man we can send the request to the app and then we get output
# from our side we will basically give a input and that will capture input and model will give the output

def predict_api():#input given in jason format which will capture inside the data key
    data=request.json["data"]
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))##single data pt record we get
    new_data=scalar.transform(np.array(list(data.values())).reshape(1,-1))
    output=regmodel.predict(new_data) 
    print(output[0])
    return jsonify(output[0])


if __name__=="__main__":
    app.run(debug=True)
    