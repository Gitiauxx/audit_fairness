import pandas as pd
import numpy as np
from aufair import auditing_data as ad

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

pd.set_option('display.max_columns', 500)

# train and test data
train = pd.read_csv('..\\..\\..\\data\\adult_income_dataset.csv')
	
# clean features
train['workclass'] = train['workclass'].astype('category').cat.codes
train['education'] = train['education'].astype('category').cat.codes
train['occupation'] = train['occupation'].astype('category').cat.codes
train['relationship'] = train['relationship'].astype('category').cat.codes
train['marital-status'] = train['marital-status'].astype('category').cat.codes
train['income'] = train['income_bracket'].astype('category').cat.codes
train['gender'] =  train['sex'].astype('category').cat.codes
train['srace'] =  train['race'].astype('category').cat.codes
train['income'] = -2 * train.income + 1

test = pd.read_csv('..\\..\\..\\data\\adult_income_test.csv')
	
# clean features
test['workclass'] = test['workclass'].astype('category').cat.codes
test['education'] = test['education'].astype('category').cat.codes
test['occupation'] = test['occupation'].astype('category').cat.codes
test['relationship'] = test['relationship'].astype('category').cat.codes
test['marital-status'] = test['marital-status'].astype('category').cat.codes
test['income'] = test['income_bracket'].astype('category').cat.codes
test['gender'] =  test['sex'].astype('category').cat.codes
test['srace'] =  test['race'].astype('category').cat.codes
test['income'] = -2 * test.income + 1

feature_list = ['age', 'workclass', 'education',  'occupation', 
		'hours-per-week', 'capital-gain', 'education-num', 'srace']
outcome = 'income'
protected = {'sex': [' Male', ' Female'], 'race':[' Black', ' White']}

# classifier -- aggregate
train['attr'] =  1 -2* train['gender']
test['attr'] = 1 -2* test['gender']

# logistic regression
logreg = LogisticRegression()
dct = DecisionTreeClassifier(max_depth=10)
rf = RandomForestClassifier(n_estimators=100)
logreg.fit(np.array(train[feature_list]), np.array(train[outcome].ravel()))

protected = ('attr', 1)
yname = 'predict'
test_x = np.array(test[feature_list])
test['predict'] = logreg.predict(test_x)

# auditing the classifier
auditor = DecisionTreeClassifier(max_depth=20, min_samples_leaf=0.01)

feature_auditing = [ 'age', 'workclass', 'education',  'occupation', 
		'hours-per-week', 'capital-gain', 'education-num', 'srace']
for f in feature_auditing:
	test[f] = (test[f] - test[f].mean())/ np.sqrt(test[f].var())

audit = ad.detector_data(auditor, test, protected, yname, lw=0.005, niter=0)
audit.get_y()

g, g_std = audit.certify_iter(feature_auditing, 'predict',  nboot=1, balancing='MMD')
print(g)

"""
g, alpha, test_start = audit.audit_weight(feature_auditing, 'attr', 0, seed=3, conv=40)
#g, alpha, test_start = audit.audit(feature_auditing, seed=3)
test_end = test_start[(test_start.predicted == 1) & (test_start.weight > 0)]

print(test_start[test_start.attr == 0].weight.sum())
print(test_start[test_start.attr == 1].weight.sum())

print(alpha)
print(test_end[test_end.attr == 0].describe())
print(test_end[test_end.attr != 0].describe())
print((test_end.label * test_end.weight / test_end.weight.sum()).sum())
"""
        
