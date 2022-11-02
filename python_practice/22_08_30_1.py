import pandas as pd
import matplotlib.pyplot as plt

sub = pd.read_csv('data/subway.csv', encoding='cp949')
#노선번호별 승차에 대한 평균
sub.groupby(sub['노선번호']).mean()['승차']

#모든행의 0,2,3열을 노선번호 열로 그룹핑하고 합계와 평균 연산 결과 출력
sub.iloc[:,[0,2,3]].groupby('노선번호').agg(['sum','mean'])

#노선번호별 승차 데이터 수확인
sub.groupby(sub['노선번호'])['승차'].count()
#sub.groupby(sub['노선번호'])['승차'].count().size()

import seaborn as sns

titantic = sns.load_dataset('titanic')
df = titantic[['age','sex','class','fare','survived']]

#class열에는 first, second, third라는 3개의 값들이 들어있다. 이열을 기준으로 그룹객체를 생성하고 그룹별 평균을 출력
grouped = df.groupby('class').mean()

for key, group in grouped:
    print(key)
    print(len(group))  
    print(group)
    print('\n')

#class 열에는 first, second, third라는 3개의 값들이 들어 있다. 이 열을 기준으로 그룹 객체를 생성하고 'First'그룹 데이터만 출력
group1 = df.groupby('class').get_group('First')
group1.head()

#class열과 성별열을 기준으로 그룹객체를 생성하고, 각 그룹별 데이터 수와 첫 5 rows 출력
grouped_two = df.groupby(['class','sex'])
for key, group in grouped_two:
    print(key)
    print(len(group))  
    print(group)
    print('\n')

#class 열로 그룹핑하고 fare 열의 min값, max값, age열에는 mean 연산 결과 출력
grouped = df.groupby('class').agg({'fare':['min','max'],'age':'mean'})

#그룹객체 필터링, class 열로 그룹핑하고 그룹핑한 행수가 200이상인 그룹만 출력
test = df.groupby('class').filter(lambda x: len(x)>200)
print(test.head())

#class 열로 그룹핑하고 age 열의 평균이 30 미만인 그룹만 출력
test = df.groupby('class').filter(lambda x: x['age'].mean()<30)
print(test.head())

#