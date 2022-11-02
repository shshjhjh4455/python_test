from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from sklearn.linear_model import LinearRegression

diabetes = load_diabetes()
print(diabetes.data.shape, diabetes.target.shape)

df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
df['target'] = diabetes.target
df.head()

X = diabetes.data
Y = diabetes.target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

lr = LinearRegression()
lr.fit(X_train, Y_train)

y_pred = lr.predict(X_test)
print(y_pred)

print('coefficients: ', lr.coef_)
print('intercept: ', lr.intercept_)

print('Mean squared error: %.2f' % mean_squared_error(Y_test, y_pred))
print('Coefficient of determination: %.2f' % r2_score(Y_test, y_pred))

# 딥러닝으로 학습시켜서 오차가 최소가되는 각 변수의 계수, 절편 찾기(회기분석)
