# 인공신경망 라이브러리 tensorflow와 keras 기반 다중 클래스 분류

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical


df = pd.read_csv("data/iris.csv")
df.info()
df.head()

import seaborn as sns
import matplotlib.pyplot as plt

#hue 옵션은 주어진 데이터 중 어떤 카테고리를 중심으로 시각화할지 설정
sns.pairplot(df, hue='species')
plt.show()

X =df.iloc[:, :4]
y = df.iloc[:, 4]

print(X[:5])
print(y[:5])

y =pd.get_dummies(y)
print(y[:5])

import tensorflow as tf
##은닉층은 몇 층으로 할지, 은닉층 안의 노드는 몇 개로 할지에 대한 정답은 없습니다.
## 노드의 수와 은닉층의 개수를 바꾸어 보면서 더 좋은 정확도가 나오는지 확인해보세요

model = Sequential()
model.add(Dense(12, input_dim=4, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X, y, epochs=100, batch_size=10)

