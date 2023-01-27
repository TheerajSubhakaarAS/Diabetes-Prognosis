import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd


app=Flask(__name__)
model=pickle.load(open('diabetes_model_SVM.sav','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    output = model.predict(data)
    return jsonify(output[0])

@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input=(np.array(data).reshape(1,-1))
    output = model.predict(final_input)[0]
    if int(output)==1:
        prediction = "The person is prone to diabetic"
    else:
        prediction = "The person not is prone to diabetic"
    return render_template("result.html",prediction = prediction)
    #return render_template("home.html",prediction_text="This person is categorised as : {}".format(output))


if __name__=="__main__":
    app.run(debug=True)