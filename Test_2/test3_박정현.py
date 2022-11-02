# price_apt.csv 파일을 사용, 파이썬을 활용하여 공공데이터포털의 국토교통부 아파트매매 실거래자료 데이터를 데이터프레임에 저장하고 시각화합니다

import pandas as pd
import matplotlib.pyplot as plt
import os

if os.name == "posix":
    plt.rc("font", family="AppleGothic")


else:
    plt.rc("font", family="Malgun Gothic")


# 1. 데이터프레임 생성
df = pd.read_csv("data/price_apt.csv")

# 보증금(만원), 월세(만원), 건축년도, 시군구, 단지명, 전용면적(㎡) 데이터를 수집합니다
df = df[["보증금(만원)", "월세(만원)", "건축년도", "시군구", "단지명", "전용면적(㎡)"]]
print(df.head(3))

# 보증금(만원), 월세(만원), 건축년도 데이터의 결측치와 문자를 제거하고 정수형으로 변환합니다
df["보증금(만원)"] = df["보증금(만원)"].str.replace(",", "").str.replace("-", "0").astype(int)
df["월세(만원)"] = df["월세(만원)"].astype(int)
df["건축년도"] = df["건축년도"].astype(int)
print(df.head(3))


# 전용면적(㎡) 을 float 타입으로 변환합니다
df["전용면적(㎡)"] = df["전용면적(㎡)"].astype(float)

# 시군구 데이터에 미아동이 포함된 데이터를 추출하여 df_miadong 변수에 저장합니다
df_miadong = df[df["시군구"].str.contains("미아동")]
print(df_miadong.head(3))

# 시군구 데이터에 수유동이 포함된 데이터를 추출하여 df_miadong 변수에 저장합니다
df_suyudong = df[df["시군구"].str.contains("수유동")]

# 시군구 데이터에 번동이 포함된 데이터를 추출하여 df_miadong 변수에 저장합니다
df_bundong = df[df["시군구"].str.contains("번동")]

# 시군구 데이터에 우이동이 포함된 데이터를 추출하여 df_miadong 변수에 저장합니다
df_uyidong = df[df["시군구"].str.contains("우이동")]

# 미아동, 수유동, 번동, 우이동  4개의 각 동별로 보증금(만원)을 평균을 내서 히스토그램으로 시각화 합니다.
plt.hist(df_miadong["보증금(만원)"], bins=20, alpha=0.5, label="미아동")
plt.hist(df_suyudong["보증금(만원)"], bins=20, alpha=0.5, label="수유동")
plt.hist(df_bundong["보증금(만원)"], bins=20, alpha=0.5, label="번동")
plt.hist(df_uyidong["보증금(만원)"], bins=20, alpha=0.5, label="우이동")
plt.legend()
plt.show()

# 미아동, 수유동, 번동, 우이동  4개의 각 동별로 월세(만원)를 평균을 내서 히스토그램으로  시각화 합니다.
plt.hist(df_miadong["월세(만원)"], bins=20, alpha=0.5, label="미아동")
plt.hist(df_suyudong["월세(만원)"], bins=20, alpha=0.5, label="수유동")
plt.hist(df_bundong["월세(만원)"], bins=20, alpha=0.5, label="번동")
plt.hist(df_uyidong["월세(만원)"], bins=20, alpha=0.5, label="우이동")
plt.legend()
plt.show()

