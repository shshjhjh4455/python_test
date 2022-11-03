# breast_cancer dataset load

from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

breast_cancer = load_breast_cancer()
features = breast_cancer.data
target = breast_cancer.target

# train, test split
features_train, features_test, target_train, target_test = train_test_split(
    features, target, test_size=0.3, random_state=0
)

# DecisionTreeClassifier
decisiontree = DecisionTreeClassifier(random_state=0)
model = decisiontree.fit(features_train, target_train)

# predict
target_predicted = model.predict(features_test)

# accuracy
print("decisiontree_Accuracy:", metrics.accuracy_score(target_test, target_predicted))
# Accuracy: 0.9122807017543859

# DecisionTreeClassifier with Gini
decisiontree_gini = DecisionTreeClassifier(
    criterion="gini", max_depth=4, random_state=0
)
model_gini = decisiontree_gini.fit(features_train, target_train)

# predict
target_predicted = model_gini.predict(features_test)

# accuracy
print(
    "decisiontree_gini_Accuracy:", metrics.accuracy_score(target_test, target_predicted)
)
# decisiontree_gini_Accuracy: 0.9473684210526315

# DecisionTreeClassifier with entropy
decisiontree_entropy = DecisionTreeClassifier(criterion="entropy", random_state=0)
model_entropy = decisiontree_entropy.fit(features_train, target_train)

# predict
target_predicted = model_entropy.predict(features_test)

# accuracy
print(
    "decisiontree_entropy_Accuracy:",
    metrics.accuracy_score(target_test, target_predicted),
)
# Accuracy: 0.935672514619883

# DecisionTreeClassifier with entropy and max_depth
entropy2 = DecisionTreeClassifier(criterion="entropy", max_depth=4, random_state=0)
model_entropy = entropy2.fit(features_train, target_train)

# predict
target_predicted = model_entropy.predict(features_test)

# accuracy
print("Accuracy:", metrics.accuracy_score(target_test, target_predicted))
# Accuracy: 0.9473684210526315
