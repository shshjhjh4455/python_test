import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler


train = pd.read_csv("data/titanic_train.csv")
test = pd.read_csv("data/titanic_test.csv")
# print(train.shape)
# print(test.shape)

gender = np.zeros(len(train))
gender[train["Sex"] == "male"] = 1
gender[train["Sex"] == "female"] = 0
train["Sex"] = gender

# Embarked 결측치 처리
train["Embarked"].fillna(train["Embarked"].mode()[0], inplace=True)

# Embarked Label Encoding
embarked = np.zeros(len(train))
embarked[train["Embarked"] == "C"] = 1
embarked[train["Embarked"] == "Q"] = 2
embarked[train["Embarked"] == "S"] = 3
train["Embarked"] = embarked

# PassengerId, Name, Ticket, Cabin 특성변수 삭제
drop_features = ["PassengerId", "Name", "Ticket", "Cabin"]
train.drop(drop_features, axis=1, inplace=True)
test.drop(drop_features, axis=1, inplace=True)

# Age의 결측값을 평균으로 채움
train["Age"].fillna(train["Age"].mean(), inplace=True)
print(train.info())
y = train["Survived"]
X = train.drop("Survived", axis=1)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=1
)

scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create a model, fit it to the training data, print its accuracy,
# and predict the categories of the test data.
gbrt = GradientBoostingClassifier(random_state=0)
gbrt.fit(X_train_scaled, y_train)
print("Accuracy on training set: {}".format(gbrt.score(X_train_scaled, y_train)))
y_pred = gbrt.predict(X_test_scaled)
print("Accuracy on test set: {}".format(accuracy_score(y_test, y_pred)))
# Accuracy on training set: 0.9149277688603531
# Accuracy on test set: 0.7686567164179104

# titanic 데이터 GradientBoostClassifier의 best 파라미터 찾기
hyperparameters = {
    "loss": ["log_loss", "exponential"],
    "learning_rate": [0.01, 0.1, 0.2, 0.3],
    "n_estimators": [50, 100, 150, 200],
    "subsample": [0.1, 0.2, 0.5, 1.0],
    "max_depth": [2, 3, 4, 5],
}

grid = GridSearchCV(
    estimator=GradientBoostingClassifier(random_state=1),
    param_grid=hyperparameters,
    scoring="roc_auc",
    n_jobs=-1,
    cv=5,
)
grid.fit(X_train, y_train)
print(grid.best_score_)
print(grid.best_params_)
print(grid.best_estimator_)
# 0.8765370421329418
# {'learning_rate': 0.1, 'loss': 'exponential', 'max_depth': 2, 'n_estimators': 100, 'subsample': 0.5}
# GradientBoostingClassifier(loss='exponential', max_depth=2, random_state=1,
#                            subsample=0.5)
