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

@app.route("/predict",methods=["POST"])
def predict():
    data=[float(x)for x in request.form.values()]#whatever values we are filling in that form we will be able to capture it bcoz all the request will be present in request object
    final_input=scalar.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output=regmodel.predict(final_input)[0]
    return render_template("home.html",prediction_text="The House price prediction is {}".format(output))#reder is v.imp in flask

if __name__=="__main__":
    app.run(debug=True)
    