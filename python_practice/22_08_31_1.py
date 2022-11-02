DataUrl = 'https://raw.githubusercontent.com/Datamanim/pandas/main/timeTest.csv'

import pandas as pd
import matplotlib.pyplot as plt

#데이터를 로드하고 각 열의 데이터 타입을 확인
df = pd.read_csv(DataUrl)
print(df.dtypes)

#Yr_Mo_Dy 열을 pandas에서 인식할 수 있는 datetime64 타입으로 변환
df['Yr_Mo_Dy'] = pd.to_datetime(df['Yr_Mo_Dy'])
print(df.dtypes)

#Yr_Mo_Dy 열에 존재하는 년도의 유일값을 모두 출력
print(df['Yr_Mo_Dy'].dt.year.unique())

#Yr_Mo_Dy 열에 년도가 2061년 이상의 경우에는 모두 잘못된 데이터이다. 해당 경우의 값은 100을 빼서 새롭게 날짜를 Yr_Mo_Dy 컬럼에 저장한다. 
def fix_century(x):
    import datetime
    year= x.year -100 if x.year > 2060 else x.year
    return pd.to_datetime(datetime.date(year,x.month,x.day))

df['Yr_Mo_Dy']=df['Yr_Mo_Dy'].apply(fix_century)

#년도별 각 컬럼의 평균값을 구한다.
df_mean = df.groupby(df['Yr_Mo_Dy'].dt.year).mean()
print(df_mean)

#weekday 컬럼을 만들고 요일별로 매핑한다.
df['weekday'] = df['Yr_Mo_Dy'].dt.weekday
print(df['weekday'])

#weekday 컬럼을 기준으로 주말이면 1 평일이면 0의 값을 가지는 WeekCheck 컬럼을 만든다.
df['WeekCheck'] = df['weekday'].map({'Saturday':1, 'Sunday':1, 'Monday':0, 'Tuesday':0, 'Wednesday':0, 'Thursday':0, 'Friday':0})
print(df['WeekCheck'])

#년도, 일자 상관없이 모든 컬럼의 각 달의 평균값을 구한다.
df_mean_month = df.groupby([df['Yr_Mo_Dy'].dt.month]).mean()
print(df_mean_month)

#모든 결측치는 컬럼기준 직전의 값으로 대체하고 첫번째 행에 결측치가 있을경우 뒤에있는 값으로 대체한다.
df_mean_month.fillna(method='ffill', inplace=True)
print(df_mean_month)

#년도 -월을 기준으로 모든 컬럼의 평균값을 구한다.
df_mean_year_month = df.groupby([df['Yr_Mo_Dy'].dt.year, df['Yr_Mo_Dy'].dt.month]).mean()
print(df_mean_year_month)
