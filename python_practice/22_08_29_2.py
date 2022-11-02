import pandas as pd
import xml.etree.ElementTree as ET
import sys
import matplotlib.pyplot as plt
if sys.version_info[0] == 3:
    from urllib.request import urlopen
else:
    from urllib import urlopen

url = "https://www.w3schools.com/xml/cd_catalog.xml"
response = urlopen(url).read()
xtree = ET.fromstring(response)
print(xtree)

rows = []
# iterate through each node of the tree
for node in xtree:
    n_title = node.find("TITLE").text
    n_artist = node.find("ARTIST").text
    n_country = node.find("COUNTRY").text
    n_company = node.find("COMPANY").text
    n_price = node.find("PRICE").text
    n_year = node.find("YEAR").text
    rows.append({"title": n_title,
    "artist": n_artist,
    "country": n_country,
    "company": n_company,
    "price": n_price,
    "year": n_year})
print(rows)

#XML text data를 dict로 저장된 list를 pandas DataFrame으로 변환
columns = ["title", "artist", "country", "company", "price", "year"]
catalog_cd_df = pd.DataFrame(rows, columns = columns)
catalog_cd_df.head(10)
#df.dtypes 로 각 칼럼의 데이터 형태를 확인 - 문자열 객체(string object)
print(catalog_cd_df.dtypes)
# astype()을 이용하여 칼럼 중에서 price는 float64, year는 int32로 변환
import numpy as np
catalog_cd_df = catalog_cd_df.astype({'price': np.float, 'year': int})
print(catalog_cd_df.dtypes)
country_mean = catalog_cd_df.groupby('country').price.mean()
country_mean
country_mean_df = pd.DataFrame(country_mean).reset_index()
import seaborn as sns
sns.barplot(x='country', y='price', data=country_mean_df)
plt.show()