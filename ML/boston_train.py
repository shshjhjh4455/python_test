from sklearn.datasets import load_boston
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

boston = load_boston()

# 모든 특성(열) 변수에 결측값 확인
print(boston.data.shape, boston.target.shape)
print(boston.feature_names)
print(boston.DESCR)

# 데이터 프레임으로 변환
df = pd.DataFrame(boston.data, columns=boston.feature_names)
df["target"] = boston.target
# df.head()

# 집값 평균과 특성변수들의 상관성 확인(heatmap)
plt.figure(figsize=(10, 10))
sns.heatmap(df.corr(), annot=True, fmt=".2f")
plt.show()

# LSTAT, RM, CRIM, NOX 특성변수들만 X변수에 저장
X = df[["LSTAT", "RM", "CRIM", "NOX"]]

# y변수에는 집값 평균(target) 저장
y = df["target"]

# 학습셋과 테스트셋으로 나누기
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# LinearRegression으로 학습셋 훈련, 생성된 모델의 절편과 계수 출력
lr = LinearRegression()
lr.fit(X_train, y_train)
print(lr.intercept_)
print(lr.coef_)
# 절편:  40.995595172164336
# 계수:  [-0.64401299  4.63744745 -0.20496083 -17.76661123]

# test셋 예측된 값과 실제값의 평가 지표 RMSE와 R2 출력
y_pred = lr.predict(X_test)
print("RMSE: ", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2: ", r2_score(y_test, y_pred))
# RMSE:  5.500939819468743
# R2:  0.5678770071387287

# regplot을 이용하여 추세선 출력
sns.regplot(x=y_test, y=y_pred, fit_reg=True)
plt.show()

# 잔차의 분포를 밀도 히스토그램 그래프로 출력 bins=15
sns.distplot(y_test - y_pred, bins=15)
plt.show()

# 학습 데이터를 degree=2인 다항 특성 데이터 생성
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)

# LinearRegression으로 학습셋 훈련, 생성된 모델의 절편과 계수 출력
lr = LinearRegression()
lr.fit(X_train_poly, y_train)
print(lr.intercept_)
print(lr.coef_)
# 절편:  40.995595172164336
# 계수:  [-0.64401299  4.63744745 -0.20496083 -17.76661123]

# 데스트 데이터 셋도 다항 특성 데이터로 변환 후 예측
X_test_poly = poly.transform(X_test)
y_pred = lr.predict(X_test_poly)

# test셋 예측된 값과 실제값의 평가 지표 RMSE와 R2 출력
print("RMSE: ", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2: ", r2_score(y_test, y_pred))
# RMSE:  4.209985010944817
# R2:  0.7468985186675552

# regplot을 이용하여 추세선 출력
sns.regplot(x=y_test, y=y_pred, fit_reg=True)
plt.show()

