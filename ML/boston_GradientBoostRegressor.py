import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_boston
from sklearn.model_selection import GridSearchCV

# boston 데이터셋 로드
boston = load_boston()

# boston 데이터셋 DataFrame 변환
bostonDF = pd.DataFrame(boston.data, columns=boston.feature_names)
bostonDF["PRICE"] = boston.target
print("Boston 데이터셋 크기: ", bostonDF.shape)
bostonDF.head()

# 학습과 테스트 데이터 세트 분리
X_data = bostonDF.iloc[:, :-1]
y_target = bostonDF.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(
    X_data, y_target, test_size=0.3, random_state=0
)

# boston house price prediction using GradientBoostRegressor
# GBM Regressor 생성
gbm_reg = GradientBoostingRegressor(random_state=0)
gbm_reg.fit(X_train, y_train)
gbm_pred = gbm_reg.predict(X_test)
gbm_mse = mean_squared_error(y_test, gbm_pred)
gbm_rmse = np.sqrt(gbm_mse)
print("MSE: {0:.3f}, RMSE: {1:.3F}".format(gbm_mse, gbm_rmse))

# 피처별 중요도 시각화
def plot_feature_importances(model):
    n_features = X_data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align="center")
    plt.yticks(np.arange(n_features), X_data.columns)
    plt.xlabel("Feature Importances")
    plt.ylabel("Features")
    plt.ylim(-1, n_features)


plot_feature_importances(gbm_reg)
plt.show()

# find best parameters using GridSearchCV

params = {
    "loss": [
        "squared_error",
        "absolute_error",
        "huber",
        "quantile",
        "deviance",
        "exponential",
    ],
    "n_estimators": [50, 100, 200, 500],
    "learning_rate": [0.05, 0.1, 0.15, 0.2],
    "max_depth": [3, 5, 7, 9],
}

# GridSearchCV 객체 생성
grid_cv = GridSearchCV(gbm_reg, param_grid=params, cv=2, verbose=1)
grid_cv.fit(X_data, y_target)

print("GridSearchCV 최적 하이퍼 파라미터: ", grid_cv.best_params_)
print("GridSearchCV 최고 예측 정확도: {0:.4f}".format(grid_cv.best_score_))
# GridSearchCV 최적 하이퍼 파라미터:  {'learning_rate': 0.1, 'loss': 'absolute_error', 'max_depth': 3, 'n_estimators': 500}
# GridSearchCV 최고 예측 정확도: 0.7628

# GridSearchCV를 이용해 최적화된 estimator로 예측 및 평가 수행
gbm_reg = GradientBoostingRegressor(
    learning_rate=0.1, loss="absolute_error", max_depth=3, n_estimators=500
)
gbm_reg.fit(X_train, y_train)
gbm_pred = gbm_reg.predict(X_test)
gbm_mse = mean_squared_error(y_test, gbm_pred)
gbm_rmse = np.sqrt(gbm_mse)
print("MSE: {0:.3f}, RMSE: {1:.3F}".format(gbm_mse, gbm_rmse))


# 피처별 중요도 시각화
plot_feature_importances(gbm_reg)
plt.show()
