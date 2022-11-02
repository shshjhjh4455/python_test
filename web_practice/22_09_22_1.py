# using BeautifulSoup "https://library.gabia.com/" 웹크롤링하여 포스터의 제목과 링크 추출한다.

from bs4 import BeautifulSoup
import requests
import pandas as pd
import selenium

page = requests.get("https://library.gabia.com/")
soup = BeautifulSoup(page.content, 'html.parser')

# 포스터들의 제목과 링크를 추출한다.
#eg-55-post-id-11645 > div > div.esg-entry-content.eg-grant-content.esg-notalone > div.esg-content.eg-post-11645.eg-grant-element-0-a > a > span

title = []
links = []
elements = soup.select('div.esg-content.eg-grant-element-0-a')

for index, element in enumerate(elements):
    title.append(element.text)
    links.append(element.find('a')['href'])
    
# 추출한 데이터를 데이터프레임으로 만든다.
df = pd.DataFrame({'title':title, 'links':links})
print(df)

# 데이터프레임을 csv 파일로 저장한다.
df.to_csv('web_practice/22_09_22_1.csv', index=False)

