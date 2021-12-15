#----to do previously (save the model and save the scaler/transformation to use in input data)
#model.save(“student_model.h5”)
#import joblib
#joblib.dump(scaler,'student_scaler.pkl')

from flask import Flask, render_template, session, redirect, url_for, session, request, jsonify
from flask_wtf import FlaskForm
from wtforms import StringField,SubmitField
from wtforms.validators import NumberRange
import numpy as np 
from tensorflow.keras.models import load_model
import pandas as pd
import joblib 
from sklearn import preprocessing

        
def return_prediction(model,unpickled_test_df,sample_json):
  #  Change this to 15 Inputs
  INTAKE_COLLEGE_EXPERIENCE = sample_json['INTAKE_COLLEGE_EXPERIENCE']
  PRIMARY_PROGRAM_CODE= sample_json['PRIMARY_PROGRAM_CODE']
  SCHOOL_CODE= sample_json['SCHOOL_CODE']
  PROGRAM_SEMESTERS= sample_json['PROGRAM_SEMESTERS']
  TOTAL_PROGRAM_SEMESTERS= sample_json['TOTAL_PROGRAM_SEMESTERS']
  MAILING_POSTAL_CODE_GROUP_3= sample_json['MAILING_POSTAL_CODE_GROUP_3']
  GENDER= sample_json['GENDER']
  DISABILITY_IND= sample_json['DISABILITY_IND']
  CURRENT_STAY_STATUS= sample_json['CURRENT_STAY_STATUS']
  ACADEMIC_PERFORMANCE= sample_json['ACADEMIC_PERFORMANCE']
  FIRST_YEAR_PERSISTENCE_COUNT= sample_json['FIRST_YEAR_PERSISTENCE_COUNT']
  ENGLISH_TEST_SCORE= sample_json['ENGLISH_TEST_SCORE']
  AGE_GROUP_LONG_NAME= sample_json['AGE_GROUP_LONG_NAME']
  APPLICANT_CATEGORY_NAME= sample_json['APPLICANT_CATEGORY_NAME']


  dict = {'INTAKE COLLEGE EXPERIENCE':INTAKE_COLLEGE_EXPERIENCE,
  'PRIMARY PROGRAM CODE':PRIMARY_PROGRAM_CODE,
  'SCHOOL CODE':SCHOOL_CODE,
  'PROGRAM SEMESTERS':PROGRAM_SEMESTERS,
  'TOTAL PROGRAM SEMESTERS':TOTAL_PROGRAM_SEMESTERS,
  'MAILING POSTAL CODE GROUP 3':MAILING_POSTAL_CODE_GROUP_3,
  'GENDER':GENDER,
  'DISABILITY IND':DISABILITY_IND,
  'CURRENT STAY STATUS':CURRENT_STAY_STATUS,
  'ACADEMIC PERFORMANCE':ACADEMIC_PERFORMANCE,
  'FIRST YEAR PERSISTENCE COUNT':FIRST_YEAR_PERSISTENCE_COUNT,
  'ENGLISH TEST SCORE':ENGLISH_TEST_SCORE,
  'AGE GROUP LONG NAME':AGE_GROUP_LONG_NAME,
  'APPLICANT CATEGORY NAME':APPLICANT_CATEGORY_NAME}

  unpickled_test_df = unpickled_test_df.append(dict, ignore_index = True)
  unpickled_test_df = unpickled_test_df.drop(['SUCCESS LEVEL'],axis=1)
  #categorical data

  categorical_cols = ['INTAKE COLLEGE EXPERIENCE','PRIMARY PROGRAM CODE','SCHOOL CODE','MAILING POSTAL CODE GROUP 3','GENDER','DISABILITY IND','CURRENT STAY STATUS',
                      'APPLICANT CATEGORY NAME'] 

  unpickled_test_df = pd.get_dummies(unpickled_test_df, columns = categorical_cols)

  unpickled_test_df['AGE GROUP LONG NAME'] = unpickled_test_df['AGE GROUP LONG NAME'].apply(lambda x: ['0 to 18', '19 to 20','21 to 25', '26 to 30','31 to 35', '36 to 40', '41 to 50'].index(x))
  unpickled_test_df['ACADEMIC PERFORMANCE'] = unpickled_test_df['ACADEMIC PERFORMANCE'].apply(lambda x: ['ZZ - Unknown','DF - Poor','C - Satisfactory','AB - Good'].index(x))

  x = unpickled_test_df.values #returns a numpy array
  min_max_scaler = preprocessing.MinMaxScaler()
  x_scaled = min_max_scaler.fit_transform(x)
  group3_hype_norm = pd.DataFrame(x_scaled)
  group3_hype_norm.columns = unpickled_test_df.columns

  group3_hype_norm = group3_hype_norm.iloc[-1:]
  
  #predict with saved model
  student_status = np.array(['unsuccessful','successful'])

  student_status_ind = (model.predict(group3_hype_norm) > 0.5).astype(int)

  #return successful or unsuccessful
  return str(student_status[student_status_ind.tolist()[0]].tolist()[0])


app = Flask(__name__, template_folder="templete")
# Configure a secret SECRET_KEY
app.config['SECRET_KEY'] = 'someRandomKey'

# Loading the model and scaler
student_model = load_model('Group3_HYPE.h5')
# Read Data Frame For one hot encodeing
unpickled_test_df = pd.read_pickle("./group3_hype_test_df.pkl")
# Now create a WTForm Class
class StudentForm(FlaskForm):

  INTAKE_COLLEGE_EXPERIENCE = StringField('INTAKE_COLLEGE_EXPERIENCE')
  PRIMARY_PROGRAM_CODE = StringField('PRIMARY_PROGRAM_CODE')
  SCHOOL_CODE = StringField('SCHOOL_CODE')
  PROGRAM_SEMESTERS = StringField('PROGRAM_SEMESTERS')
  TOTAL_PROGRAM_SEMESTERS = StringField('TOTAL_PROGRAM_SEMESTERS')
  MAILING_POSTAL_CODE_GROUP_3 = StringField('MAILING_POSTAL_CODE_GROUP_3')
  GENDER = StringField('GENDER')
  DISABILITY_IND = StringField('DISABILITY_IND')
  CURRENT_STAY_STATUS = StringField('CURRENT_STAY_STATUS')
  ACADEMIC_PERFORMANCE = StringField('ACADEMIC_PERFORMANCE')
  SUCCESS_LEVEL= StringField('SUCCESS_LEVEL')
  FIRST_YEAR_PERSISTENCE_COUNT= StringField('FIRST_YEAR_PERSISTENCE_COUNT')
  ENGLISH_TEST_SCORE= StringField('ENGLISH_TEST_SCORE')
  AGE_GROUP_LONG_NAME= StringField('AGE_GROUP_LONG_NAME')
  APPLICANT_CATEGORY_NAME= StringField('APPLICANT_CATEGORY_NAME')
  submit = SubmitField('Analyze')

@app.route('/', methods=['GET'])
def index():
  # Create instance of the form.
  form = StudentForm()
  # If the form is valid on submission
  return render_template('home.html', form=form)

@app.route('/', methods=['POST'])
def index_data():
  # Create instance of the form.

  #Defining content dictionary
  content = {}

  content['INTAKE_COLLEGE_EXPERIENCE'] = str(request.form['INTAKE_COLLEGE_EXPERIENCE'])
  content['PRIMARY_PROGRAM_CODE'] = int(request.form['PRIMARY_PROGRAM_CODE'])
  content['SCHOOL_CODE'] = str(request.form['SCHOOL_CODE'])
  content['PROGRAM_SEMESTERS'] = int(request.form['PROGRAM_SEMESTERS'])
  content['TOTAL_PROGRAM_SEMESTERS'] = int(request.form['TOTAL_PROGRAM_SEMESTERS'])
  content['MAILING_POSTAL_CODE_GROUP_3'] = str(request.form['MAILING_POSTAL_CODE_GROUP_3'])
  content['GENDER'] = str(request.form['GENDER'])
  content['DISABILITY_IND'] = str(request.form['DISABILITY_IND'])
  content['CURRENT_STAY_STATUS'] = str(request.form['CURRENT_STAY_STATUS'])
  content['ACADEMIC_PERFORMANCE'] = str(request.form['ACADEMIC_PERFORMANCE'])
  content['FIRST_YEAR_PERSISTENCE_COUNT'] = int(request.form['FIRST_YEAR_PERSISTENCE_COUNT'])
  content['ENGLISH_TEST_SCORE'] = float(request.form['ENGLISH_TEST_SCORE'])
  content['AGE_GROUP_LONG_NAME'] = str(request.form['AGE_GROUP_LONG_NAME'])
  content['APPLICANT_CATEGORY_NAME'] = str(request.form['APPLICANT_CATEGORY_NAME'])
  print(content)
  results = return_prediction(model=student_model,unpickled_test_df=unpickled_test_df,sample_json=content)

  return render_template('prediction.html',results=results)

@app.route('/prediction')
def prediction():

  content = request.json
  print(content)

  results = return_prediction(model=student_model,unpickled_test_df=unpickled_test_df,sample_json=content)


  return jsonify({"results":results})

if __name__ == '__main__':
 app.run(debug=True)