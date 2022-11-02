from unittest import result
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import KFold, cross_val_score, cross_validate
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

rf =  RandomForestClassifier(random_state=1)

k_fold = KFold(n_splits=10, shuffle=True, random_state=1)

# 평가지표를 객체로 생성
scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score),
    'recall': make_scorer(recall_score),
    'f1': make_scorer(f1_score),
    'roc_auc': make_scorer(roc_auc_score)
}

result = cross_validate(rf, X, y, cv=k_fold, scoring=scoring)
print(result)

# 평균값을 출력
print('accuracy: ', result['test_accuracy'].mean())
print('precision: ', result['test_precision'].mean())
print('recall: ', result['test_recall'].mean())
print('f1: ', result['test_f1'].mean())
print('roc_auc: ', result['test_roc_auc'].mean())

