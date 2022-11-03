from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

diabetes = load_diabetes()
features = diabetes.data
target = diabetes.target


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
# decisiontree_Accuracy: 0.0

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
# decisiontree_gini_Accuracy: 0.007518796992481203

# DecisionTreeClassifier with entropy
decisiontree_entropy = DecisionTreeClassifier(
    criterion="entropy", max_depth=3, random_state=0
)
model_entropy = decisiontree_entropy.fit(features_train, target_train)

# predict
target_predicted = model_entropy.predict(features_test)

# accuracy
print(
    "decisiontree_entropy_Accuracy:",
    metrics.accuracy_score(target_test, target_predicted),
)
# decisiontree_entropy_Accuracy: 0.015037593984962405
