import pandas as pd
from time import time
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, explained_variance_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.model_selection import GridSearchCV

boston = load_boston()

# 특성 데이터 정규화
scaler = StandardScaler()
boston.data = scaler.fit_transform(boston.data)

regressors = [LinearRegression, Ridge, Lasso, ElasticNet, DecisionTreeRegressor, RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, XGBRegressor, LGBMRegressor, CatBoostRegressor]
for model in regressors:
    print(model.__name__)
    start = time()
    X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size=0.2, random_state=1)
    model = model()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print('time: ', time() - start)
    print('MAE: ', mean_absolute_error(y_test, y_pred))
    print('R2: ', r2_score(y_test, y_pred))
    print('Explained Variance Score: ', explained_variance_score(y_test, y_pred))
    print('----------------------------------------')


#  찾은 최적 모델GradientBoostingRegressor()의 하이퍼파라미터 최적값 찾기
X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size=0.2, random_state=1)
model = GradientBoostingRegressor()
params = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.1, 0.05, 0.01],
    'max_depth': [4, 6],
    'min_samples_leaf': [3, 5, 9, 17],
    'max_features': [1.0, 0.3, 0.1]
}
grid_cv = GridSearchCV(model, param_grid=params, cv=5, n_jobs=-1)
grid_cv.fit(X_train, y_train)
print('최적 하이퍼파라미터: ', grid_cv.best_params_)
print('최고 예측 정확도: ', grid_cv.best_score_)
print('최고 예측 모델: ', grid_cv.best_estimator_)
y_pred = grid_cv.predict(X_test)
print('MAE: ', mean_absolute_error(y_test, y_pred))
print('R2: ', r2_score(y_test, y_pred))
print('Explained Variance Score: ', explained_variance_score(y_test, y_pred))



