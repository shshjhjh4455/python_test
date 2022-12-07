# 이직여부 판단 데이터를 로드하여 전처리를 수행하고 분류 모델을 생성하여 검증 데이터로 분류 예측 성능을 출력하는  프로그램을 작성
# 1. 데이터를 로드하여 전처리를 수행

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import sklearn.model_selection
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# 데이터 로드
x_train = pd.read_csv(
    "https://raw.githubusercontent.com/Datamanim/datarepo/main/HRdata/X_train.csv"
)
y_train = pd.read_csv(
    "https://raw.githubusercontent.com/Datamanim/datarepo/main/HRdata/y_train.csv"
)
x_test = pd.read_csv(
    "https://raw.githubusercontent.com/Datamanim/datarepo/main/HRdata/X_test.csv"
)

print(x_train["gender"].unique())  # Other
print(x_train["enrolled_university"].unique())  # no_enrollment
print(x_train["education_level"].unique())  # 최빈값
print(x_train["major_discipline"].unique())  # 최빈값
print(x_train["experience"].unique())  # 0
print(x_train["company_size"].unique())  # <10
print(x_train["company_type"].unique())  # Other
print(x_train["last_new_job"].unique())  # never

# 데이터 전처리
# experience 경력 변수의 값이 '>20'인경우 '20+'로, '<1'인경우 '1-' 변환
x_train["experience"] = x_train["experience"].replace(">20", "20+")
x_train["experience"] = x_train["experience"].replace("<1", "1-")
x_test["experience"] = x_test["experience"].replace(">20", "20+")
x_test["experience"] = x_test["experience"].replace("<1", "1-")

# company_size 회사규모(직원수) 변수의 값이 '<10'인 경우 '10-'로 변환
x_train["company_size"] = x_train["company_size"].replace("<10", "10-")
x_test["company_size"] = x_test["company_size"].replace("<10", "10-")

# last_new_job 이전 직장 변수의 값이 '>4'이면 '4+'로 변환
x_train["last_new_job"] = x_train["last_new_job"].replace(">4", "4+")
x_test["last_new_job"] = x_test["last_new_job"].replace(">4", "4+")

# gender 성별의 결측치를 'Other'로 채우기
x_train["gender"] = x_train["gender"].fillna("Other")
x_test["gender"] = x_test["gender"].fillna("Other")

# enrolled_university  등록된 대학 의 결측치를 'no_enrollment'로 채우기
x_train["enrolled_university"] = x_train["enrolled_university"].fillna("no_enrollment")
x_test["enrolled_university"] = x_test["enrolled_university"].fillna("no_enrollment")

# education_level 교육수준을STEM'변수의 최빈값으로 채우기
x_train["major_discipline"] = x_train["major_discipline"].fillna("STEM")
x_test["major_discipline"] = x_test["major_discipline"].fillna("STEM")

# major_discipline  전공을 'Graduate'변수의 최빈값으로 채우기
x_train["education_level"] = x_train["education_level"].fillna("Graduate")
x_test["education_level"] = x_test["education_level"].fillna("Graduate")

# experience 경력을 결측치를 0로 채우기
x_train["experience"] = x_train["experience"].fillna(0)
x_test["experience"] = x_test["experience"].fillna(0)

# company_size 회사규모(직원수)를 결측치를 '10-'로 채우기
x_train["company_size"] = x_train["company_size"].fillna("10-")
x_test["company_size"] = x_test["company_size"].fillna("10-")

# company_type 고용주 유형을 결측치를 'other'로 채우기
x_train["company_type"] = x_train["company_type"].fillna("Other")
x_test["company_type"] = x_test["company_type"].fillna("Other")

# last_new_job 이전 직장과 현재 직장의 연도 차이를 'never' 로 채우기
x_train["last_new_job"] = x_train["last_new_job"].fillna("never")
x_test["last_new_job"] = x_test["last_new_job"].fillna("never")

# 컬럼 삭제 enrollee_id(고유 id), city(도시 코드) 삭제
x_test_id = x_test["enrollee_id"]
x_train.drop(columns=["enrollee_id", "city"], inplace=True)
x_test.drop(columns=["enrollee_id", "city"], inplace=True)
y_train.drop(columns=["enrollee_id"], inplace=True)

# 표준화로 스케일링 처리하기
ss = StandardScaler()
x_train["training_hours"] = ss.fit_transform(x_train[["training_hours"]])
x_test["training_hours"] = ss.fit_transform(x_test[["training_hours"]])

# 범주 데이터를 One-hot encoding으로 변환
x_train = pd.get_dummies(x_train)
x_test = pd.get_dummies(x_test)

# 학습, 검증 데이터셋으로 분리
X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(
    x_train, y_train, test_size=0.22, random_state=42
)

# RandomForest로 이직 여부 분류 예측
rf = RandomForestClassifier()
rf.fit(X_TRAIN, Y_TRAIN)
Y_TEST_prob = rf.predict_proba(X_TEST)[:, 1]
y_test_prob = rf.predict_proba(x_test)[:, 1]

# 검증 데이터의 정확률 출력
print(accuracy_score(Y_TEST, rf.predict(X_TEST)))

# 검증 데이터의 AUC 출력
print(roc_auc_score(Y_TEST, Y_TEST_prob))

# roc_auc 시각화
fpr, tpr, thresholds = roc_curve(Y_TEST, Y_TEST_prob)
plt.plot(fpr, tpr, label="ROC Curve")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.show()

# XGBoost로 이직 여부 분류 예측
xgb = XGBClassifier()
xgb.fit(X_TRAIN, Y_TRAIN)
Y_TEST_prob = xgb.predict_proba(X_TEST)[:, 1]
y_test_prob = xgb.predict_proba(x_test)[:, 1]

# 검증 데이터의 정확률 출력
print(accuracy_score(Y_TEST, xgb.predict(X_TEST)))

# 검증 데이터의 AUC 출력
print(roc_auc_score(Y_TEST, Y_TEST_prob))

# roc_auc 시각화
fpr, tpr, thresholds = roc_curve(Y_TEST, Y_TEST_prob)
plt.plot(fpr, tpr, label="ROC Curve")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.show()
