from wordcloud import WordCloud
from turtle import width
from konlpy.tag import Okt
from collections import Counter
import matplotlib.pyplot as plt

with open('data/대한민국헌법.txt', 'r', encoding='utf-8') as f:
    text = f.read()

okt = Okt()
nouns = okt.nouns(text) # 명사 추출
words = [word for word in nouns if len(word) > 1] 
c = Counter(words)
print(c)

wc = WordCloud(font_path='malgun',width=400,height=400,scale=2.0,max_font_size=250).generate_from_frequencies(c)
plt.figure()
plt.imshow(wc, interpolation='bilinear')
wc.to_file('헌법_워드클라우드.png')