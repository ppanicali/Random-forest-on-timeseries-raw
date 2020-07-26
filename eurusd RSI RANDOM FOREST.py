# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 21:12:47 2020

@author: 
    
    RSI eur usd random tree 50 cross??
"""


##CARICA I DATI DELLA TIME SERIE
import pandas as pd
import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
#from sklearn.lda import LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import confusion_matrix
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
#from sklearn.qda import QDA
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

#carica dati dal file in df
df = pd.read_csv('C:\\Users\\Paolo\\Documents\\PYTHON EURUSD RSI RANDOM TREE//EURUSD1H_RSI.csv',sep=";",decimal=",", engine='python')
#trasforma la colonna Date in Datetime con il suo formato
df['Date'] =  pd.to_datetime(df['Date'], format='%d/%m/%Y %H:%M:%S')
#mostra tipo dati delle colonne di df
df.dtypes
df=df.sort_values(by=['Date'])
#avendo ordinato per data tuttavia l indice generale risulta alterato, percui viene resettato con resetindex
df=df.reset_index(drop=True)

df=df.drop(['Open'],axis=1)
#df=df.drop(['High'],axis=1)
#df=df.drop(['Low'],axis=1)
df=df.drop(['Volume'],axis=1)

# df['Close1']=df['Close'].shift(1)
# df['Close2']=df['Close'].shift(2)

# df['RSI20_1']=df['RSI20'].shift(1)
# df['RSI20_2']=df['RSI20'].shift(2)

# df['RSI40_1']=df['RSI40'].shift(1)
# df['RSI40_2']=df['RSI40'].shift(2)
# df['RSI40_3']=df['RSI40'].shift(3)
# df['RSI40_4']=df['RSI40'].shift(4)

df['RSI80_1']=df['RSI80'].shift(1)
df['RSI80_2']=df['RSI80'].shift(2)
df['RSI80_3']=df['RSI80'].shift(3)
df['RSI80_4']=df['RSI80'].shift(4)

df=df[10:len(df)]

df=df.reset_index(drop=True)

# LABELLING 

a_stop = 0.15
a_profit = 0.25

for index in range(0,len(df)-72):    
    current_price = df.at[index , 'Close']
    stop_loss = current_price-(current_price/100*a_stop)
    take_profit = current_price+(current_price/100*a_profit)
    counter=1
    long=1
            
    while long==1:  
        high_l = df.at[index + counter , 'High']
        low_l = df.at[index + counter , 'Low']
                
        if low_l < stop_loss:
            df.loc[index,"Label"]=0
            long=0
            break
        if high_l >= take_profit:
            df.loc[index,"Label"]=1
            long=0
            break
        counter = counter + 1
        

df=df.drop(['High'],axis=1)
df=df.drop(['Low'],axis=1) 
df=df.drop(['Date'],axis=1) 
df=df.drop(['Close'],axis=1) 

df=df.drop(['RSI20'],axis=1)   
df=df.drop(['RSI40'],axis=1)


     

df2=df[20000:50000]    


X = df2.drop(['Label'],axis=1)
y = df2["Label"]

start_test=40000
# Create training and test sets
X_train = X[X.index < start_test]
X_test = X[X.index >= start_test]
y_train = y[y.index < start_test]
y_test = y[y.index >= start_test]
# Create the (parametrised) models
print("Hit Rates/Confusion Matrices:\n")
models = [
         ("RF", RandomForestRegressor(n_estimators=1000, 
             max_depth=None, min_samples_split=2,
             min_samples_leaf=1, max_features='auto',
             bootstrap=True, oob_score=False, n_jobs=1,
             random_state=None, verbose=0
             )
             )]
# Iterate through the models


for m in models:
    # Train each of the models on the training set
    m[1].fit(X_train, y_train)
    


    # Make an array of predictions on the test set
    pred = m[1].predict(X_test)
    # Output the hit-rate and the confusion matrix for each model
    print("%s:\n%0.3f" % (m[0], m[1].score(X_test, y_test)))
    print("%s\n" % confusion_matrix(pred, y_test))  