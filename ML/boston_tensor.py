from sklearn.datasets import load_boston
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

boston = load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = pd.DataFrame(boston.target)

model = Sequential()
model.add(Dense(64, input_dim=13, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics=['mae'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
monitor_val_acc = EarlyStopping(monitor='val_loss', patience=15, mode='auto')

ModelCheckpoint = ModelCheckpoint('best_model.h5', save_best_only=True)

model.fit(X_train, y_train, validation_data=(X_test, y_test), callbacks=[monitor_val_acc, ModelCheckpoint], epochs=500)

plt.figure(figsize=(15, 8))
plt.xlabel('Epochs')
plt.ylabel('mse')
plt.plot(model.history.history['loss'])
plt.plot(model.history.history['val_loss'])
plt.show()

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print('학습 데이터셋의 평균 주택 가격:', round(y.mean()*1000,2))
print('학습 데이터셋의 MAE:', round(mae*1000,2))
print('학습 데이터셋의 MSE:', round(mse,2))

best_model = tf.keras.models.load_model('best_model.h5')
y_pred = best_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print('학습 데이터셋의 평균 주택 가격:', round(y.mean()*1000,2))
print('best model의 MAE:', round(mae*1000,2))
print('best model의 MSE:', round(mse,2))



