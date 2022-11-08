import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud


imdb_train = pd.read_csv("data/IMDB_train.csv")
imdb_test = pd.read_csv("data/IMDB_test.csv")

# 긍정 리뷰와 부정 리뷰 개수 출력
print(imdb_train["label"].value_counts())

# label=0 부정 리뷰, label=1 긍정 리뷰
# 부정 리뷰, 긍정 리뷰 수 비율 시각화 (countplot)
sns.set(style="whitegrid", palette="muted", font_scale=1.5)
f = sns.countplot(x="label", data=imdb_train)
f.set_title("Sentiment Distribution")
f.set_xticklabels(["Negative", "Positive"])
plt.xlabel("")
plt.show()

# 빈도수가 높은 단어 시각화 (wordcloud)
# 긍정 리뷰
pos_reviews = imdb_train[imdb_train["label"] == 1]
pos_reviews = pos_reviews["text"]
pos_reviews = pos_reviews.values
pos_reviews = " ".join(pos_reviews)
pos_reviews = pos_reviews.lower()
pos_reviews = pos_reviews.replace("<br />", " ")

# 부정 리뷰
neg_reviews = imdb_train[imdb_train["label"] == 0]
neg_reviews = neg_reviews["text"]
neg_reviews = neg_reviews.values
neg_reviews = " ".join(neg_reviews)
neg_reviews = neg_reviews.lower()
neg_reviews = neg_reviews.replace("<br />", " ")

# wordcloud
pos_wc = WordCloud(background_color="white", max_words=2000, width=1600, height=800)
pos_wc.generate(pos_reviews)
neg_wc = WordCloud(background_color="white", max_words=2000, width=1600, height=800)
neg_wc.generate(neg_reviews)

# 시각화
plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
plt.imshow(pos_wc, interpolation="bilinear")
plt.axis("off")
plt.title("Positive Reviews", fontsize=20)
plt.subplot(1, 2, 2)
plt.imshow(neg_wc, interpolation="bilinear")
plt.axis("off")
plt.title("Negative Reviews", fontsize=20)
plt.show()

# naive bayes 모델 학습
# CountVectorizer
cv = CountVectorizer(binary=True)
cv.fit(imdb_train["text"])
X = cv.transform(imdb_train["text"])
X_test = cv.transform(imdb_test["text"])

# MultinomialNB
target = imdb_train["label"].values
clf = MultinomialNB()
clf.fit(X, target)
print("Accuracy: ", accuracy_score(imdb_test["label"].values, clf.predict(X_test)))
# Accuracy:  0.8568
