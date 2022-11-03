import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import pandas as pd

fish = pd.read_csv("data/fish.csv")
fish.info()

# 예측할 품종의 종류(label) 확인
print(pd.unique(fish["Species"]))

# 데이터의 feature(특성)과 label(정답)을 분리
fish_input = fish.iloc[:, 1:].to_numpy()
fish_target = fish.iloc[:, 1].to_numpy()
print(fish_input[:5])
print(fish_target[:5])

# 학습, 테스트 데이터셋 분할
X_train, X_test, Y_train, Y_test = train_test_split(
    fish_input, fish_target, test_size=0.25, random_state=42
)

# feature 데이터 정규화
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
train_scaled = ss.fit_transform(X_train)
test_scaled = ss.fit_transform(X_test)
print(pd.DataFrame(train_scaled).describe())

# Y(Species) = w1*(Weight) + w2*(Length) + w3*(Diagonal) + w4*(Height) + w5*(Width) + b
# 가중치 w1, w2, w3, w4, w5와 b(bias 편향)은 로지스틱 함수에서 비용함수를 통해 최적의 값을 훈련과정을 통해 구함
# 다항 로지스틱회귀는 시그모이드 함수 대신 softmax함수 사용
# softmax함수는 예측 값을 0~1 사이의 값으로 주여주고 모든 입력값의 합이 1이 되도록 줄여줌

lr = LogisticRegression(C=20, max_iter=1000)
lr.fit(train_scaled, Y_train)

print(lr.score(train_scaled, Y_train))  # 성능 평가
print(lr.score(test_scaled, Y_test))
print(lr.predict(test_scaled[:5]))  # 테스트 데이터 상위 5개만 예측
proba = lr.predict_proba(test_scaled[:5])  ##테스트 데이터 상위 5개 예측 확률
print(lr.classes_)  # 모델의 예측 클래스 종류
print(np.round(proba, decimals=3))

# 회귀계수와 bias 확인
print(lr.coef_.shape)
print(lr.intercept_.shape)
# 클래스가 7개이므로 클래스별 분류에 적용되는 회귀계수와 bias값이 모두 다름
for i in range(7):
    print(lr.coef_[i], lr.intercept_[i])

# 분류 클래스별 확률 기반으로 Y값을 출력해주는 함수
decision = lr.decision_function(test_scaled[:5])
print(np.round(decision, decimals=2))
