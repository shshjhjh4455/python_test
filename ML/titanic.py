# titanic file load

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
from sklearn.metrics import accuracy_score
import graphviz


# load data
df_train = pd.read_csv("data/titanic_train.csv")
df_test = pd.read_csv("data/titanic_test.csv")

sns.countplot(x="Survived", data=df_train)

sns.countplot(x="Sex", data=df_train)

sns.catplot(x="Survived", col="Sex", kind="count", data=df_train)

df_train.groupby(["Sex"]).Survived.sum()

print(
    df_train[df_train.Sex == "female"].Survived.sum()
    / df_train[df_train.Sex == "female"].Survived.count()
)
print(
    df_train[df_train.Sex == "male"].Survived.sum()
    / df_train[df_train.Sex == "male"].Survived.count()
)

sns.catplot(x="Survived", col="Embarked", kind="count", data=df_train)

df_train_drop = df_train.dropna()
sns.pairplot(df_train_drop, hue="Survived")

survived_train = df_train.Survived
data = pd.concat([df_train.drop(["Survived"], axis=1), df_test])
data.info()

data["Age"] = data.Age.fillna(data.Age.median())
data["Fare"] = data.Fare.fillna(data.Fare.median())
data.info()

data = pd.get_dummies(data, columns=["Sex"], drop_first=True)
data.head()

data = data[["Sex_male", "Fare", "Age", "Pclass", "SibSp"]]
data.head()

data_train = data.iloc[:891]
data_test = data.iloc[891:]
X = data_train.values
test = data_test.values
y = survived_train.values

clf = DecisionTreeClassifier()
clf.fit(X, y)

dot_data = export_graphviz(
    clf,
    out_file=None,
    feature_names=data_train.columns.values,
    class_names=["Survived", "Not Survived"],
    filled=True,
    rounded=True,
    special_characters=True,
)
graph = graphviz.Source(dot_data)
graph

y_pred = clf.predict(test)
df_test["Survived"] = y_pred
clf.score(X, y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42, stratify=y
)

dep = np.arange(1, 9)
train_accuracy = np.empty(len(dep))
test_accuracy = np.empty(len(dep))

for i, k in enumerate(dep):
    clf = DecisionTreeClassifier(max_depth=k)
    clf.fit(X_train, y_train)
    train_accuracy[i] = clf.score(X_train, y_train)
    test_accuracy[i] = clf.score(X_test, y_test)

plt.plot(dep, test_accuracy, label="Testing Accuracy")
plt.plot(dep, train_accuracy, label="Training Accuracy")
plt.legend()
plt.xlabel("Depth of tree")
plt.ylabel("Accuracy")
plt.show()

clf = DecisionTreeClassifier(max_depth=6)
clf.fit(X, y)
clf.score(X, y)


dot_data = export_graphviz(
    clf,
    out_file=None,
    feature_names=data_train.columns.values,
    class_names=["Survived", "Not Survived"],
    filled=True,
    rounded=True,
    special_characters=True,
)
graph = graphviz.Source(dot_data)
graph

y_pred = clf.predict(test)
df_test["Survived"] = y_pred
df_test[["PassengerId", "Survived"]].to_csv(
    "data/titanic_depth6_decisiontree.csv", index=False
)
