#Classification of 20 News Group
#사이킷 런에서는 Twenty Newsgroups이라고 불리는 20개의 다른 주제를 가진 18,846개의 뉴스그룹 데이터를 제공
#훈련 데이터(11,314개)와 테스트 데이터(7,532개)로 분류되어 있음

from sklearn.datasets import fetch_20newsgroups
newsdata=fetch_20newsgroups(subset='train')
print(newsdata.keys())

print(newsdata.target_names) #20개의 카테고리의 이름
#target에는 총 0부터 19까지의 숫자가 저장되어 있음
#첫번째 샘플이 어떤 카테고리에 속하는지 확인
print(newsdata.target[0]) # 7
print(newsdata.target_names[7])
##첫번째 샘플의 기사 확인
print(newsdata.data[0])  #토큰화 안되어 있음
 
# Count Vectorization 피처 벡터화 변환 진행
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X_train = cv.fit_transform(newsdata.data)
print(X_train.shape) # 샘플 개수와 단어의 개수 

# 가중치주고 성능 향상하여 학습데이터 생성
from sklearn.feature_extraction.text import TfidfTransformer
tfidf = TfidfTransformer()
tfidf_x_train = tfidf.fit_transform(X_train)
print(tfidf_x_train.shape)

# 다항 분포 나이브 베이즈 모델 학습
from sklearn.naive_bayes import MultinomialNB 
mod = MultinomialNB()
mod.fit(tfidf_x_train, newsdata.target)

# 학습된 모델로 테스트 데이터 news group 분류
newsdata_test = fetch_20newsgroups(subset='test',shuffle=True)
X_test = cv.transform(newsdata_test.data)
print(X_test.shape)

tfidf_x_test = tfidf.transform(X_test)
print(tfidf_x_test.shape)

from sklearn.metrics import  accuracy_score
y_pred = mod.predict(tfidf_x_test)
print("정확도:", accuracy_score(newsdata_test.target, y_pred))
