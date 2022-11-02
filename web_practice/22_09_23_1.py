# 네이버 검색 API 예제 - 블로그 검색
import requests 
from urllib.parse import urlparse

client_id = " "
client_secret = " "

searchWorld =  "수리남"
url = "https://openapi.naver.com/v1/search/blog?query=" + searchWorld # JSON 결과
# url = "https://openapi.naver.com/v1/search/blog.xml?query=" + encText # XML 결과
response = requests.get(urlparse(url).geturl(), headers = {"X-Naver-Client-Id" : client_id, "X-Naver-Client-Secret": client_secret})
print(response.json())