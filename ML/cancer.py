from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.2, random_state=1)

model = LogisticRegression()
model.fit(X_train, y_train)
print('LogisticRegression')
print('train score: ', model.score(X_train, y_train))
print('test score: ', model.score(X_test, y_test))
print('----------------------------------------')

# LogisticRegression 모델의 하이퍼파라미터 최적값 찾기
params = {
    'penalty': ['l1', 'l2'],
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['liblinear', 'saga']
}
grid_cv = GridSearchCV(model, param_grid=params, cv=5, n_jobs=-1)
grid_cv.fit(X_train, y_train)
print('최적 하이퍼파라미터: ', grid_cv.best_params_)
print('최고 예측 정확도: ', grid_cv.best_score_)
print('최고 예측 모델: ', grid_cv.best_estimator_)
print('----------------------------------------')

model = RandomForestRegressor()
model.fit(X_train, y_train)
print('RandomForestRegressor')
print('train score: ', model.score(X_train, y_train))
print('test score: ', model.score(X_test, y_test))
print('----------------------------------------')

# RandomForestRegressor 모델의 하이퍼파라미터 최적값 찾기
params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [6, 8, 10, 12],
    'min_samples_leaf': [8, 12, 18],
    'min_samples_split': [8, 16, 20]
}
grid_cv = GridSearchCV(model, param_grid=params, cv=5, n_jobs=-1)
grid_cv.fit(X_train, y_train)
print('최적 하이퍼파라미터: ', grid_cv.best_params_)
print('최고 예측 정확도: ', grid_cv.best_score_)
print('최고 예측 모델: ', grid_cv.best_estimator_)
print('----------------------------------------')

# LogisticRegression
# train score:  0.9648351648351648
# test score:  0.956140350877193
# ----------------------------------------

# RandomForestRegressor
# train score:  0.9999999999999999
# test score:  0.8991228070175439
# ----------------------------------------





