import streamlit as st 
import pandas as pd
import numpy as np
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
from sqlalchemy import create_engine




import joblib
import pickle

model = pickle.load(open('DT.pkl','rb'))
impute = joblib.load('meanimpute')
winsor = joblib.load('winsor')
minmax = joblib.load('minmax')
encoding = joblib.load('encoding')


def predict_Downtime(data, user, pw, db):

    conn_string = ("postgresql+psycopg2://{user}:{pw}@localhost/{db}"
                  .format(user ="postgres",
                         pw = 1234,
                         db = "Project"))
    db = create_engine(conn_string)
    conn = db.connect()

    data_new = data
    clean1 = pd.DataFrame(impute.transform(data_new), columns = data_new.select_dtypes(exclude = ['object']).columns)
    clean2 = pd.DataFrame(winsor.transform(clean1))
    clean3 = pd.DataFrame(minmax.transform(clean2))
    clean4 = pd.DataFrame(encoding.transform(data_new),columns = data_new.select_dtypes(include = ['object']).columns)
    clean_data = pd.concat([clean3, clean4], axis = 1, ignore_index = True)
    prediction = pd.DataFrame(model.predict(clean_data), columns = ['Downtime'])
    final_data = pd.concat([prediction, data_new], axis = 1)
    final_data.to_sql('Predicted_by_streamlit', con = conn, if_exists = 'replace', chunksize = 1000, index= False)

    return final_data



def main():
    

    st.title("Makino Machine Downtime prediction")
    st.sidebar.title("Makino Machine Downtime prediction")

    # st.radio('Type of Cab you want to Book', options=['Mini', 'Sedan', 'XL', 'Premium', 'Rental'])
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Makino Machine Downtime prediction </h2>
    </div>
    
    """
    st.markdown(html_temp, unsafe_allow_html = True)
    st.text("")
    

    uploadedFile = st.sidebar.file_uploader("Choose a file", type=['csv','xlsx'], accept_multiple_files=False, key="fileUploader")
    if uploadedFile is not None :
        try:

            data = pd.read_csv(uploadedFile)
        except:
                try:
                    data = pd.read_excel(uploadedFile)
                except:      
                    data = pd.DataFrame()
        
        
    else:
        st.sidebar.warning("You need to upload a CSV or an Excel file.")
    
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <p style="color:white;text-align:center;">Add DataBase Credientials </p>
    </div>
    """
    st.sidebar.markdown(html_temp, unsafe_allow_html = True)
            
    user = st.sidebar.text_input("user", "Type Here")
    pw = st.sidebar.text_input("password", "Type Here")
    db = st.sidebar.text_input("database", "Type Here")
    
    result = ""
    
    if st.button("Predict"):
        result = predict_Downtime(data, user, pw, db)
        #st.dataframe(result) or
        #st.table(result.style.set_properties(**{'background-color': 'white','color': 'black'}))
                           
     
        cm = sns.light_palette("blue", as_cmap = True)
        st.table(result.style.background_gradient(cmap=cm).set_precision(2))

if __name__=='__main__':
    main()


