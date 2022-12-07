import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
from collections import Counter
from konlpy.tag import Mecab
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/bab2min/corpus/master/sentiment/naver_shopping.txt",
    filename="ratings_total.txt",
)

total_data = pd.read_table("ratings_total.txt", names=["ratings", "reviews"])
# 총 20만개의 샘플이 존재
# print('전체 리뷰 개수 :',len(total_data)) # 전체 리뷰 개수 출력

# 평점이 4, 5인 리뷰에는 레이블 1을, 평점이 1, 2인 리뷰에는 레이블 0을 부여합니다.
total_data = total_data[total_data["ratings"] != 3]
total_data["label"] = np.select([total_data["ratings"] > 3], [1], default=0)
# print(total_data[:5])

# reviews열에서 중복을 제외한 샘플의 수를 카운트하시오
# print(total_data['reviews'].nunique()) # 199908

# NULL 값 유무를 확인합니다.
print(total_data.isnull().values.any())  # False

# 훈련 데이터와 테스트 데이터를 3:1 비율로 분리
train_data, test_data = train_test_split(total_data, test_size=0.25, random_state=42)
# print('훈련용 리뷰 개수 :',len(train_data)) # 훈련용 리뷰 개수 출력

# 훈련 데이터의 레이블의 분포를 확인하시오. 그래프 출력
train_data["label"].value_counts().plot(kind="bar")
# plt.show()

# 테스트 데이터에 대해서도 정규 표현식을 사용하여 한글을 제외하고 모두 제거하시오
test_data["reviews"] = test_data["reviews"].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "")
# print(test_data[:5])
# 형태소 분석기 Mecab을 사용하여 토큰화 작업을 수행
mecab = Mecab()
train_data["tokenized"] = train_data["reviews"].apply(mecab.morphs)
test_data["tokenized"] = test_data["reviews"].apply(mecab.morphs)
# print(train_data[:5])

# 훈련 데이터와 테스트 데이터에 불용어를 지정하여 필요없는 토큰들은 제거합니다
stopwords = [
    "도",
    "는",
    "다",
    "의",
    "가",
    "이",
    "은",
    "한",
    "에",
    "하",
    "고",
    "을",
    "를",
    "인",
    "듯",
    "과",
    "와",
    "네",
    "들",
    "듯",
    "지",
    "임",
    "게",
]

train_data["tokenized"] = train_data["tokenized"].apply(
    lambda x: [item for item in x if item not in stopwords]
)

test_data["tokenized"] = test_data["tokenized"].apply(
    lambda x: [item for item in x if item not in stopwords]
)

# 긍정 리뷰에는 주로 어떤 단어들이 많이 등장하는지 확인
# 긍정 리뷰에 대해서만 추출
positive = train_data[train_data["label"] == 1]
# 긍정 리뷰에 대해서만 토큰화된 단어들을 모두 추출
positive_words = np.hstack(positive["tokenized"].values)
# 긍정 리뷰 빈도수 순으로 정렬
positive_word_count = Counter(positive_words)
# print(positive_word_count.most_common(20))

# 부정 리뷰에는 주로 어떤 단어들이 많이 등장하는지 확인
# 부정 리뷰에 대해서만 추출
negative = train_data[train_data["label"] == 0]
# 부정 리뷰에 대해서만 토큰화된 단어들을 모두 추출
nagtive_words = np.hstack(negative["tokenized"].values)
# 부정 리뷰 빈도수 순으로 정렬
negative_word_count = Counter(nagtive_words)
# print(negative_word_count.most_common(20))

# 텍스트를 숫자로 처리할 수 있도록 훈련 데이터와 테스트 데이터에 정수 인코딩을 수행
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_data["tokenized"])
# print(tokenizer.word_index)

# 훈련 데이터와 테스트 데이터에 정수 인코딩을 수행하고, 훈련 데이터에 대해서 단어 집합(vocaburary)을 만듭니다.

# 훈련 데이터에 대해서 정수 인코딩을 수행
train_sequences = tokenizer.texts_to_sequences(train_data["tokenized"])
# print(train_sequences[:5])

# 테스트 데이터에 대해서 정수 인코딩을 수행
test_sequences = tokenizer.texts_to_sequences(test_data["tokenized"])
# print(test_sequences[:5])

# 훈련 데이터에 대해서 vocaburary 생성
word_vocab = tokenizer.word_index
# print(word_vocab)

# 서로 다른 길이의 샘플들의 길이를 동일하게 맞춰주는 패딩 작업을 진행합니다.
# 훈련 데이터에 대해서 패딩 작업을 수행
train_inputs = pad_sequences(train_sequences, padding="post")
# print(train_inputs[:5])

# 테스트 데이터에 대해서 패딩 작업을 수행
test_inputs = pad_sequences(test_sequences, padding="post")
# print(test_inputs[:5])





