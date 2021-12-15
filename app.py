#----to do previously (save the model and save the scaler/transformation to use in input data)
#model.save(“student_model.h5”)
#import joblib
#joblib.dump(scaler,'student_scaler.pkl')

from flask import Flask, render_template, session, redirect, url_for, session
from flask_wtf import FlaskForm
from wtforms import TextField,SubmitField
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
  SUCCESS_LEVEL= sample_json['SUCCESS_LEVEL']
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
  'SUCCESS LEVEL':SUCCESS_LEVEL,
  'FIRST YEAR PERSISTENCE COUNT':FIRST_YEAR_PERSISTENCE_COUNT,
  'ENGLISH TEST SCORE':ENGLISH_TEST_SCORE,
  'AGE GROUP LONG NAME':AGE_GROUP_LONG_NAME,
  'APPLICANT CATEGORY NAME':APPLICANT_CATEGORY_NAME}

  unpickled_test_df = unpickled_test_df.append(dict, ignore_index = True)

  #categorical data

  categorical_cols = ['INTAKE COLLEGE EXPERIENCE','PRIMARY PROGRAM CODE','SCHOOL CODE','MAILING POSTAL CODE GROUP 3','GENDER','DISABILITY IND','CURRENT STAY STATUS',
                      'APPLICANT CATEGORY NAME'] 

  unpickled_test_df = pd.get_dummies(unpickled_test_df, columns = categorical_cols)

  unpickled_test_df['AGE GROUP LONG NAME'] = unpickled_test_df['AGE GROUP LONG NAME'].apply(lambda x: ['0 to 18', '19 to 20','21 to 25', '26 to 30','31 to 35', '36 to 40', '41 to 50'].index(x))
  unpickled_test_df['SUCCESS LEVEL'] = unpickled_test_df['SUCCESS LEVEL'].apply(lambda x: ['Unsuccessful', 'Successful'].index(x))
  unpickled_test_df['ACADEMIC PERFORMANCE'] = unpickled_test_df['ACADEMIC PERFORMANCE'].apply(lambda x: ['ZZ - Unknown','DF - Poor','C - Satisfactory','AB - Good'].index(x))

  x = unpickled_test_df.values #returns a numpy array
  min_max_scaler = preprocessing.MinMaxScaler()
  x_scaled = min_max_scaler.fit_transform(x)
  group3_hype_norm = pd.DataFrame(x_scaled)
  group3_hype_norm.columns = unpickled_test_df.columns

  group3_hype_norm = group3_hype_norm.iloc[-1:]
  
  #predict with saved model
  student_status = np.array(['successful', 'unsuccessful'])

  student_status_ind = (model.predict(group3_hype_norm) > 0.5).astype(int)

  #return successful or unsuccessful
  return student_status[student_status_ind][0]


app = Flask(__name__)
# Configure a secret SECRET_KEY
#app.config['SECRET_KEY'] = 'someRandomKey'

# Loading the model and scaler
student_model = load_model('Group3_HYPE.h5')
# Read Data Frame For one hot encodeing
unpickled_test_df = pd.read_pickle("./group3_hype_test_df.pkl")
# Now create a WTForm Class
class StudentForm(FlaskForm):

  INTAKE_COLLEGE_EXPERIENCE = TextField('INTAKE_COLLEGE_EXPERIENCE')
  PRIMARY_PROGRAM_CODE = TextField('PRIMARY_PROGRAM_CODE')
  SCHOOL_CODE = TextField('SCHOOL_CODE')
  PROGRAM_SEMESTERS = TextField('PROGRAM_SEMESTERS')
  TOTAL_PROGRAM_SEMESTERS = TextField('TOTAL_PROGRAM_SEMESTERS')
  MAILING_POSTAL_CODE_GROUP_3 = TextField('MAILING_POSTAL_CODE_GROUP_3')
  GENDER = TextField('GENDER')
  DISABILITY_IND = TextField('DISABILITY_IND')
  CURRENT_STAY_STATUS = TextField('CURRENT_STAY_STATUS')
  ACADEMIC_PERFORMANCE = TextField('ACADEMIC_PERFORMANCE')
  SUCCESS_LEVEL= TextField('SUCCESS_LEVEL')
  FIRST_YEAR_PERSISTENCE_COUNT= TextField('FIRST_YEAR_PERSISTENCE_COUNT')
  ENGLISH_TEST_SCORE= TextField('ENGLISH_TEST_SCORE')
  AGE_GROUP_LONG_NAME= TextField('AGE_GROUP_LONG_NAME')
  APPLICANT_CATEGORY_NAME= TextField('APPLICANT_CATEGORY_NAME')
  submit = SubmitField('Analyze')

@app.route('/', methods=['GET', 'POST'])
def index():
  # Create instance of the form.
  form = StudentForm()
  # If the form is valid on submission
  if form.validate_on_submit():
    # Grab the data from the input on the form.
    
    session['INTAKE_COLLEGE_EXPERIENCE'] = form.INTAKE_COLLEGE_EXPERIENCE.data
    session['PRIMARY_PROGRAM_CODE'] = form.PRIMARY_PROGRAM_CODE.data
    session['SCHOOL_CODE'] = form.SCHOOL_CODE.data
    session['PROGRAM_SEMESTERS'] = form.PROGRAM_SEMESTERS.data
    session['TOTAL_PROGRAM_SEMESTERS'] = form.TOTAL_PROGRAM_SEMESTERS.data
    session['MAILING_POSTAL_CODE_GROUP_3'] = form.MAILING_POSTAL_CODE_GROUP_3.data
    session['GENDER'] = form.GENDER.data
    session['DISABILITY_IND'] = form.DISABILITY_IND.data
    session['CURRENT_STAY_STATUS'] = form.CURRENT_STAY_STATUS.data
    session['ACADEMIC_PERFORMANCE'] = form.ACADEMIC_PERFORMANCE.data
    session['SUCCESS_LEVEL'] = form.SUCCESS_LEVEL.data
    session['FIRST_YEAR_PERSISTENCE_COUNT'] = form.FIRST_YEAR_PERSISTENCE_COUNT.data
    session['ENGLISH_TEST_SCORE'] = form.ENGLISH_TEST_SCORE.data
    session['AGE_GROUP_LONG_NAME'] = form.AGE_GROUP_LONG_NAME.data
    session['APPLICANT_CATEGORY_NAME'] = form.APPLICANT_CATEGORY_NAME.data

    return redirect(url_for('prediction'))
  return render_template('home.html', form=form)

@app.route('/prediction')
def prediction():
  #Defining content dictionary
  content = {}

  content['INTAKE_COLLEGE_EXPERIENCE'] = str(session['INTAKE_COLLEGE_EXPERIENCE'])
  content['PRIMARY_PROGRAM_CODE'] = int(session['PRIMARY_PROGRAM_CODE'])
  content['SCHOOL_CODE'] = str(session['SCHOOL_CODE'])
  content['PROGRAM_SEMESTERS'] = int(session['PROGRAM_SEMESTERS'])
  content['TOTAL_PROGRAM_SEMESTERS'] = int(session['TOTAL_PROGRAM_SEMESTERS'])
  content['MAILING_POSTAL_CODE_GROUP_3'] = str(session['MAILING_POSTAL_CODE_GROUP_3'])
  content['GENDER'] = str(session['GENDER'])
  content['DISABILITY_IND'] = str(session['DISABILITY_IND'])
  content['CURRENT_STAY_STATUS'] = str(session['CURRENT_STAY_STATUS'])
  content['ACADEMIC_PERFORMANCE'] = str(session['ACADEMIC_PERFORMANCE'])
  content['SUCCESS_LEVEL'] = str(session['SUCCESS_LEVEL'])
  content['FIRST_YEAR_PERSISTENCE_COUNT'] = int(session['FIRST_YEAR_PERSISTENCE_COUNT'])
  content['ENGLISH_TEST_SCORE'] = float(session['ENGLISH_TEST_SCORE'])
  content['AGE_GROUP_LONG_NAME'] = str(session['AGE_GROUP_LONG_NAME'])
  content['APPLICANT_CATEGORY_NAME'] = str(session['APPLICANT_CATEGORY_NAME'])

  results = return_prediction(model=student_model,unpickled_test_df=unpickled_test_df,sample_json=content)

  return render_template('prediction.html',results=results)

if __name__ == '__main__':
 app.run(debug=True)