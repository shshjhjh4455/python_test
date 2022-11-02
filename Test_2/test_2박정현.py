# 1. 네이버에 '서울 날씨' 를 검색하면 나오는 오늘의 날씨 정보에서
# 아래의 내용을 크롤링하는 코드를 완성하시오
# https://search.naver.com/search.naver?sm=top_hty&fbm=0&ie=utf8&query=%EC%84%9C%EC%9A%B8%EB%82%A0%EC%94%A8

import requests
from bs4 import BeautifulSoup
import pandas as pd
from pprint import pprint

url = "https://search.naver.com/search.naver?query=서울날씨"
html = requests.get(url).text
soup = BeautifulSoup(html, "html.parser")

# temperature(/html/body/div[3]/div[2]/div/div[1]/section[1]/div[1]/div[2]/div[1]/div[1]/div/div[2]/div/div[1]/div[1]/div[2]/strong/text()) 수집
temperature = soup.find("div", {"class": "todaytemp"})

# text(어제보다 ) 수집
text = soup.find("span", {"class": "compare"})

# 데이터프레임 생성. 컬럼명은 temperature, text
df = pd.DataFrame([{"temperature": temperature, "text": text}])

# 데이터프레임의 각 열이름과 타입을 출력하시오
print(df.dtypes)

# 데이터프레임의 첫 3개의 레코드와 마지막 3개의 레코드를 출력하시오
print(df.head(3))
print(df.tail(3))

