from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from feature_engine.outliers import Winsorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV


from sklearn.tree import DecisionTreeClassifier as DT,plot_tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score 
from sklearn import tree



import joblib
import pickle

model = pickle.load(open('DT.pkl','rb'))
impute = joblib.load('meanimpute')
winsor = joblib.load('winsor')
minmax = joblib.load('minmax')
encoding = joblib.load('encoding')

# Connecting to sql by creating sqlachemy engine
from sqlalchemy import create_engine

conn_string = ("postgresql+psycopg2://{user}:{pw}@localhost/{db}"
              .format(user ="postgres",
                     pw = 1234,
                     db = "Project"))
db = create_engine(conn_string)
conn = db.connect()

            
#define flask
app = Flask(__name__)



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/success', methods = ['POST'])
def success():
    if request.method == 'POST' :
        f = request.files['file']
        data_new = pd.read_csv(f)
        #clean5 = data_new.drop(['Date','Machine_ID'], axis =1, inplace = True)
        clean1 = pd.DataFrame(impute.transform(data_new), columns = data_new.select_dtypes(exclude = ['object']).columns)
        clean2 = pd.DataFrame(winsor.transform(clean1))
        clean3 = pd.DataFrame(minmax.transform(clean2))
        clean4 = pd.DataFrame(encoding.transform(data_new),columns = data_new.select_dtypes(include = ['object']).columns)
        clean_data = pd.concat([clean3, clean4], axis = 1, ignore_index = True)
        prediction = pd.DataFrame(model.predict(clean_data), columns = ['Downtime'])
        final_data = pd.concat([prediction, data_new], axis = 1)
        final_data.to_sql('Predicted_by_flask', con = conn, if_exists = 'replace', chunksize = 1000, index= False)
       
        return render_template("new.html", Y = final_data.to_html(justify = 'center'))

if __name__=='__main__':
    app.run(debug = True)
