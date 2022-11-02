import json
import urllib
from urllib.request import Request, urlopen
import pandas as pd
import sys


#forbiden error 없애기
urlTicker = Request("http://api.nobelprize.org/v1/prize.json", headers={'User-Agent': 'Mozilla/5.0'})
readTicker = urlopen(urlTicker).read()
jsonTicker = json.loads(readTicker)

