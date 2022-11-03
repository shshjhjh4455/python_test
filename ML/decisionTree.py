from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import datasets

iris = datasets.load_iris()  # 데이터 로드
features = iris.data
target = iris.target

decisiontree = DecisionTreeClassifier(random_state=0)  # 결정 트리 분류기 객체 생성
model = decisiontree.fit(features, target)  # 모델 훈련

observation = [[5, 4, 3, 2]]  # New 샘플 데이터
model.predict(observation)  # 샘플 데이터의 클래스 예측
model.predict_proba(observation)  # 세 개의 클래스에 대한 예측 확률을 확인


from sklearn.tree import export_graphviz
import graphviz

# 시각화
dot_data = export_graphviz(
    model,
    out_file=None,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True,
    rounded=True,
    special_characters=True,
)
graph = graphviz.Source(dot_data)
graph


# 엔트로피를 사용해 결정 트리 분류기를 훈련합니다.
decisiontree_entropy = DecisionTreeClassifier(criterion="entropy", random_state=0)
model_entropy = decisiontree_entropy.fit(features, target)  # 모델 훈련
model.predict(observation)  # 샘플 데이터의 클래스 예측
model.predict_proba(observation)  # 세 개의 클래스에 대한 예측 확률을 확인


# 시각화
dot_data = export_graphviz(
    decisiontree_entropy,
    out_file=None,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True,
    rounded=True,
    special_characters=True,
)
graph = graphviz.Source(dot_data)
graph


# 엔트로피를 사용해 결정 트리 분류기를 훈련 , 가지치기
entropy2 = DecisionTreeClassifier(criterion="entropy", max_depth=2, random_state=0)
model_entropy = entropy2.fit(features, target)  # 모델 훈련
model_entropy.predict(observation)  # 샘플 데이터의 클래스 예측
model_entropy.predict_proba(observation)  # 세 개의 클래스에 대한 예측 확률을 확인


# 시각화
dot_data = export_graphviz(
    entropy2,
    out_file=None,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True,
    rounded=True,
    special_characters=True,
)
graph = graphviz.Source(dot_data)
graph

# confustion matrix를 이용한 정확도 비교
from sklearn.metrics import confusion_matrix

confusion_matrix(target, model.predict(features))

confusion_matrix(target, decisiontree_entropy.predict(features))

confusion_matrix(target, entropy2.predict(features))

# 학습/테스트 데이터 셋 분할
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, stratify=iris.target, random_state=1
)
# 3개의 클래스에 속하는 데이터를 고루고루 선택되도록 층화추출 파라미터  stratify
entropy3 = DecisionTreeClassifier(criterion="entropy")
model_entropy3 = entropy3.fit(X_train, y_train)  # 모델 훈련
model_entropy3.predict(observation)  # 샘플 데이터의 클래스 예측
model_entropy3.predict_proba(observation)  # 세 개의 클래스에 대한 예측 확률을 확인
confusion_matrix(y_test, entropy3.predict(X_test))
# 가치치기를 안해도 오분류 발생 - train data set과 test data set의 특성이 다르기 때문에 ...
