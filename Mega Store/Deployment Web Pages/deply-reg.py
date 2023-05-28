#!/usr/bin/env python
# coding: utf-8
import streamlit as st
import joblib
import requests
import numpy as np
import pandas as pd
import pickle 
import os
from xgboost import XGBRegressor
import nltk
import re
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.metrics import mean_squared_error
reg = XGBRegressor()
reg.load_model("D:/My projects/Python/Machine Learnig/ASU/Mega Store/Saved Object/XGBReg.txt")

pickle_in = open("D:/My projects/Python/Machine Learnig/ASU/Mega Store/Saved Object/Myscale.pickle","rb")
Scaler = pickle.load(pickle_in)

with open('D:/My projects/Python/Machine Learnig/ASU/Mega Store/Saved Object/CVlist.pkl', 'rb') as pickle_file:
    CVlist = pickle.load(pickle_file)


with open('D:/My projects/Python/Machine Learnig/ASU/Mega Store/Saved Object/ridge.pickle', 'rb') as pickle_file:
    ridge = pickle.load(pickle_file)

with open('D:/My projects/Python/Machine Learnig/ASU/Mega Store/Saved Object/ptrans.pickle', 'rb') as pickle_file:
    ptrans = pickle.load(pickle_file)

cols_list=joblib.load("D:/My projects/Python/Machine Learnig/ASU/Mega Store/Saved Object/ColsList.pkl")
def dict_to_cols(df):
  MainCat=list()
  SubCat=list()
  for i in df['CategoryTree']:
    d=eval(i)
    MainCat.append(d['MainCategory'])
    SubCat.append(d['SubCategory'])
  df['MainCat']=MainCat
  df['SubCat']=SubCat
  df.drop("CategoryTree",axis=1,inplace=True)  
  return df
def States_Encoding(df,path):
  states=joblib.load(path)
  lon=list()
  lat=list()
  for i in df['State']:
    lon.append(states[i]['usa_state_longitude'])
    lat.append(states[i]['usa_state_latitude'])
  df['long']=lon
  df['lat']=lat
  df.drop(['State',"City"],axis=1,inplace=True)
  del lon
  del lat
  return df 

def Cols_To_DT(df,printout):
  cols=list(filter(lambda x:x.endswith("Date"),df.columns))
  for col in cols:
    df[col]= pd.to_datetime(df[col])
  if(printout==True):
    print(df.select_dtypes(include="datetime").columns,"Are Date Time")
  return df

def nullreplace(df):
   nullreplacer={'Row ID': 5010.902564102564,
 'Order ID': 'CA-2016-100111',
 'Order Date': '11/10/2016',
 'Ship Date': '11/10/2016',
 'Ship Mode': 'Standard Class',
 'Customer ID': 'PP-18955',
 'Customer Name': 'Paul Prost',
 'Segment': 'Consumer',
 'Country': 'United States',
 'City': 'New York City',
 'State': 'California',
 'Postal Code': 55339.397123202005,
 'Region': 'West',
 'Product ID': 'OFF-PA-10001970',
 'CategoryTree': "{'MainCategory': 'Office Supplies', 'SubCategory': 'Binders'}",
 'Product Name': 'Staples',
 'Sales': 228.21197003126545,
 'Quantity': 3.7642276422764227,
 'Discount': 0.15588492808005827,}
   for col in df.drop("Profit").columns:
      if(df[col].isnull().sum()):
         df[col].replace(nullreplacer[col],np.nan,inplace=True)
   return df
def preprocessed(df,object_cv,Scaler,cols,retY=False):
    #Date Time Prob
    #df=nullreplace(df)
    df=Cols_To_DT(df,False)
    #Category Tree
    
    df['delivery days']=df['Ship Date']-df['Order Date']
    df['delivery days']=df['delivery days'].dt.days
    
    df.drop(["Ship Date","Row ID","Country"],axis=1,inplace=True)
    
    df['Quantity']=np.log2(df['Quantity']+1)
    df['Discount']=np.log2(df['Discount']+1)
    df['Product ID']=df['Product ID'].str.split('-',expand=True)[2].astype('int64')
    df['Customer ID']=df['Customer ID'].str.split('-',expand=True)[1].astype('int64')
    df['year']=df['Order ID'].str.split('-',expand=True)[1].astype('int64')
    df['month']=df['Order Date'].dt.month.astype('int64')
    df['Order IDs']=df['Order ID'].str.split('-',expand=True)[2].astype('int64')    
    df['Order IDs']=df['Order IDs'].astype('int64')
    df.drop(['Order ID',"Order Date"],axis=1,inplace=True)
    if(retY==True):
      X=df.drop('Profit',axis=1)
      Y=df.loc[:,'Profit']
    else:
      X=df  
    
    ###Remeber To write Scaler###
    scaled_cols=X.select_dtypes(exclude=["object"]).columns
    X[scaled_cols]=Scaler.transform(X[scaled_cols])

    X=States_Encoding(X,"D:/My projects/Python/Machine Learnig/ASU/Mega Store/Saved Object/UStatesDict.pkl")
    num_feature=[0]
    
    for ghandy,i in enumerate(X.select_dtypes(include='object').columns):

        sentences = X[i].values
        cleaned_sentence = []
        for sentence in sentences:
            word = sentence.lower()  
            word = re.sub(r'^RT[\s]+', '', word)
            word = re.sub(r'#',"",word)
            word = word.split()
            word = [i for i in word if i not in set(stopwords.words('english'))]          
            word = " ".join(word)               ##joining our words back to sentences
            cleaned_sentence.append(word)       ##appending our preprocessed sentence into a new list
          

        BagofwordSs = object_cv[ghandy].transform(cleaned_sentence).toarray()
        num_feature.append(num_feature[-1]+BagofwordSs.shape[1])  
        if (ghandy==0):
          CumBagofwords = BagofwordSs
        else:
          CumBagofwords = np.concatenate([CumBagofwords,BagofwordSs],axis=1)

    X.drop(X.select_dtypes(include='object').columns,axis=1,inplace=True)
    result = pd.concat([X.reset_index(), pd.DataFrame(CumBagofwords)], axis=1)
    X=pd.DataFrame(result)   
    X.columns=X.columns.astype(str)
    new_X=X[cols]
    
    if(retY==True):
      return new_X,Y
    else:
      return new_X

df=pd.read_csv('df - Container.csv')
st.set_page_config(page_title = 'regression')
with st.container():
    right,left = st.columns(2)
    with right:
        df['Row ID']=int(st.number_input('Enter Row ID'))
        df['Order ID']=st.text_input('Enter Order ID')
        df['Order Date'] = st.date_input('Enter order date')
        df['Ship Date'] = st.date_input('Enter ship date')
        df['Ship Mode'] = st.text_input('Enter ship mode')
        df['Customer ID'] = st.text_input('Enter customer id')
        df['Customer Name']= st.text_input("Enter Name")
        df['Segment'] = st.text_input('Enter segment')
        df['Country'] = st.text_input('Enter country')
        df['City'] = st.text_input('Enter city')
        df['State'] =st.text_input('Enter state')
        df['Postal Code'] = int(st.number_input('Enter postal code')) 
        df['Region'] = st.text_input('Enter region')
        df['Product ID'] =st.text_input('Enter product ID')
        df['Product Name']= st.text_input('Enter product name')
        df['Sales'] = float(st.number_input('Enter sales'))
        df['Quantity'] = int(st.number_input('Enter quantity'))
        df['Discount'] = float(st.number_input('Enter discount'))
        df['MainCat'] = st.text_input('Enter main category')
        df['SubCat'] = st.text_input('Enter sub category')
    with left:
       path=st.text_input('Enter Path')
       model=st.selectbox("Select Model",np.array(["XGB","ridge"]))
       if st.button('Evaluate'):
        df=pd.read_csv(path)
        df=dict_to_cols(df)
        X,y=preprocessed(df,CVlist,Scaler,cols_list,retY=True)
        yscaler=MinMaxScaler()
        yscaler.fit(np.array(y).reshape(-1,1))

        if model == "ridge":
          X_trans=ptrans.fit_transform(X[cols_list[0:22]])
          y_pred=ridge.predict(X_trans)
          d=pd.DataFrame(y_pred,columns=['y_pred'])
          d['y']=y
          st.write(d)
          st.write({"acc":r2_score(y_pred,y)})
          st.write({"MSE":mean_squared_error(y_pred,y)})
          ysc=yscaler.transform(np.array(y).reshape(-1,1))
          ysc_pred=yscaler.transform(np.array(y_pred).reshape(-1,1))
          st.write({"NMSE":mean_squared_error(ysc,ysc_pred)})
          st.write({"Acc_NMSE":1-mean_squared_error(ysc,ysc_pred)})
        else:
          y_pred=reg.predict(X)
          d=pd.DataFrame(y_pred,columns=['y_pred'])
          d['y']=y
          st.write(d)
          st.write({"acc":r2_score(y_pred,y)})              
          st.write({"MSE":mean_squared_error(y_pred,y)})
          ysc=yscaler.transform(np.array(y).reshape(-1,1))
          ysc_pred=yscaler.transform(np.array(y_pred).reshape(-1,1))
          st.write({"NMSE":mean_squared_error(ysc,ysc_pred)})
          st.write({"Acc_NMSE":1-mean_squared_error(ysc,ysc_pred)})
    if st.button('Predict'):
        X=preprocessed(df,CVlist,Scaler,cols_list,retY=False)
        y_pred=reg.predict(X)
        st.write(y_pred)    
    
    


