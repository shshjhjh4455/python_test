import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler

#테스트 데이터 셋의 ROC_AUC ?

X_train = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/shipping/X_train.csv")
y_train = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/shipping/y_train.csv")
X_test = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/shipping/X_test.csv")

print(X_train.info())
# Warehouse_block,  Mode_of_Shipment , Customer_care_calls, Product_importance ,  Gender  특성 데이터는 object(문자열) 타입

#ID 열 삭제
x_train = X_train.drop('ID', axis=1)
x_test = X_test.drop('ID', axis=1)
y_train = y_train['Reached.on.Time_Y.N']

#결측치 확인
print('X_train 결측치 개수  : ' , X_train.isnull().sum())
print('X_test 결측치 개수  : ' , X_test.isnull().sum())
print('y_train 결측치 개수  : ' , y_train.isnull().sum())

#범주형 데이터 unique 값 개수 확인
X_train_nunique=[]
X_test_nunique=[]

print('____X-TRAIN____')
for col in x_train.select_dtypes(include=object) :
    print(col)
    print(x_train[col].value_counts().head())
    print(x_train[col].unique())
    print(x_train[col].nunique())
    X_train_nunique.append(x_train[col].nunique())

print('____X-TEST____')
for col in x_test.select_dtypes(include=object) :
    print(col)
    print(x_test[col].value_counts().head())
    print(x_test[col].unique())
    print(x_test[col].nunique())
    X_test_nunique.append(x_test[col].nunique())

print('X_train nunique : ' , X_train_nunique) 
print('X_test nunique : ' , X_test_nunique) 

#범주형 데이터 처리 
x_train['Customer_care_calls'] = x_train['Customer_care_calls'].replace('$7', '7').astype(int)
x_test['Customer_care_calls'] = x_test['Customer_care_calls'].replace('$7', '7').astype(int)

#수치 데이터 확인
print(x_train.describe())

from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()

#Weight_in_gms, Cost_of_the_Product 특성 변환 
x_train['Weight_in_gms'] = mms.fit_transform(x_train[['Weight_in_gms']])
x_test['Weight_in_gms'] = mms.fit_transform(x_test[['Weight_in_gms']])

x_train['Cost_of_the_Product'] = mms.fit_transform(x_train[['Cost_of_the_Product']])
x_test['Cost_of_the_Product'] = mms.fit_transform(x_test[['Cost_of_the_Product']])

x_train['Discount_offered'] = mms.fit_transform(x_train[['Discount_offered']])
x_test['Discount_offered'] = mms.fit_transform(x_test[['Discount_offered']])
print(x_train.describe())

#one-hot encoding (Warehouse_block, Mode_of_Shipment, Product_importance  , Gender ) 
x_dummies = pd.get_dummies(pd.concat([x_train, x_test]))
x_train_dummies = x_dummies[:x_train.shape[0]]
x_test_dummies = x_dummies[x_train.shape[0]:]
x_train_dummies

#학습 데이터를 학습과 검증 데이터 셋으로 분할 (test_size=0.3)
from sklearn.model_selection import train_test_split

X_TRAIN, X_VAL, Y_TRAIN, Y_VAL = train_test_split(x_train_dummies, y_train, test_size=0.3, random_state=42, stratify=y_train)

from sklearn.linear_model import  LogisticRegression
lr=LogisticRegression(C=20,max_iter=1000, random_state=42)
lr.fit(X_TRAIN, Y_TRAIN)
pred_val = lr.predict_proba(X_VAL)[:,1]

from sklearn.metrics import roc_auc_score
print(roc_auc_score(Y_VAL, pred_val))
