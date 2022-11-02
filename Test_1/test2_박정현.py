



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# train.csv 파일로부터 데이터를 로드하여 데이터프레임으로 생성하시오
train = pd.read_csv('./data/train.csv')

# 데이터프레임의 각 열이름과 타입을 출력하시오
print(train.dtypes)

# 데이터프레임의 첫 3개의 레코드와 마지막 3개의 레코드를 출력하시오
print(train.head(3))
print(train.tail(3))

# 데이터프레임의 int, float 열변수의 통계(null이 아닌 데이터 개수, 평균, 표준편차, 최소값, 1/4분위값, 중앙값, 3/4분위값, 최대값)를 출력하시오
print(train.describe())

# 데이터프레임에서 Ticket, Cabin 열을 삭제하시오
train.drop(['Ticket', 'Cabin'], axis=1, inplace=True)

# 데이터프레임에서 성별(Sex) 열의 값을 남자는 0, 여자는 1로 변환하시오
train = train.replace({'male':0, 'female':1})
print(train)

# 데이터프레임의 티켓 클래스(Pclass)에 따른 생존율(Survived)을 비교하시오
print(train[['Pclass', 'Survived']].groupby('Pclass').mean())

# # 데이터프레임에서 성별(Sex)에 따른 생존율을 비교하시오
print(train[['Sex', 'Survived']].groupby('Sex').mean())

# 생존 여부(Survived)에 따른 연령 분포를 bin 20의 히스토그램으로 시각화하시오 
survive = sns.FacetGrid(train, col='Survived')
survive.map(plt.hist, 'Age', bins=20)
plt.show()



