# Fastapi for creating api in python
from fastapi import FastAPI , Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
# we have to mention formats of input our model going to need
from pydantic import BaseModel
# load saved model
import pickle
# to convert json object to python
import json

app=FastAPI()

templates = Jinja2Templates(directory="templates")

class model_input(BaseModel):

    Pregnancies : int
    Glucose : int
    BloodPressure : int
    SkinThickness : int
    Insulin : int
    BMI : float
    DiabetesPedigreeFunction : float
    Age : int

#loading saved model
diabetes_model = pickle.load(open('diabetes_model_SVM.sav','rb'))

@app.post('/prediction_svm')
def diabetes_pred(input_parameters : model_input):

    input_data = input_parameters.json()
    input_dictionary = json.loads(input_data)

    preg = input_dictionary['Pregnancies']
    glu = input_dictionary['Glucose']
    bp = input_dictionary['BloodPressure']
    skin = input_dictionary['SkinThickness']
    insulin = input_dictionary['Insulin']
    bmi = input_dictionary['BMI']
    dpf = input_dictionary['DiabetesPedigreeFunction']
    age = input_dictionary['Age']

    input_list = [preg,glu,bp,skin,insulin,bmi,dpf,age]

    prediction = diabetes_model.predict([input_list])

    if prediction[0] == 0:
        return "The person is not Diabetic."
    else:
        return "The person is Diabetic."

