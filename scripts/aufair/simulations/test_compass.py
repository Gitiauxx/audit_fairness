import pandas as pd
import numpy as np
#from aufair import audit_differential as ad
from aufair import auditing_data as ad

from fairlearn import moments
from fairlearn import classred as red

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC

pd.set_option('display.max_columns', 500)

# load data
data = pd.read_csv('..\\..\\..\\data\\compas-scores-two-years.csv')
data = data[data.race.isin(['Caucasian', 'African-American'])]

# create categorical data for age_cat, sex, race and charge degree
data['gender'] = data.sex.astype('category').cat.codes
data['age_cat'] = data.age_cat.astype('category').cat.codes
data['charge_degree'] = data.c_charge_degree.astype('category').cat.codes
data['crace'] = data.race.astype('category').cat.codes
data['is_violent_recid'] = data.is_violent_recid.astype('category').cat.codes
data['juv_fel_count'] = data.juv_fel_count.astype('category').cat.codes
data['juv_misd_count'] = data.juv_misd_count.astype('category').cat.codes
data['count_race']  = data['priors_count'] * data['crace']
data['two_year_recid'] = 2 * data['two_year_recid'] - 1 

feature_list = ['gender', 'age_cat',  'priors_count', 'juv_fel_count', 
                'is_violent_recid', 'juv_misd_count', 'charge_degree', 'crace']

for var in feature_list: 
    data = data[~np.isnan(data[var])]

outcome = 'two_year_recid'
protected = {'race': ['Caucasian', 'African-American']}

# split train and test (70, 30)
np.random.seed(seed=1)
train = data.loc[np.random.choice(data.index, int(0.7 * len(data)))]
test = data.drop(train.index)

# classifier -- aggregate
train['attr'] =  train['crace']
test['attr'] = test['crace']

# logistic regression
logreg = LogisticRegression()
dct = DecisionTreeClassifier(max_depth=10)
rf = RandomForestClassifier(n_estimators=100)
rf.fit(np.array(train[feature_list]), np.array(train[outcome].ravel()))

protected = {'attr': 0}
data = pd.concat([train, test])
data = data.set_index(np.arange(len(data)))
data_x = np.array(data[feature_list])
data['predict'] = rf.predict(data_x)

# auditing the classifier
auditor = DecisionTreeClassifier(max_depth=10, min_samples_leaf=0.015)
audit = ad.detector_data(auditor, data, stepsize=0.05)

feature_list = ['gender', 'age_cat',  'priors_count', 'juv_fel_count', 
               'is_violent_recid', 'juv_misd_count', 'charge_degree']
balancer = LogisticRegression()
audit.get_weights(balancer, feature_list, 'attr', 0)

audit.get_y('predict', 'attr', 0)

feature_auditing = [ 'age_cat',  'priors_count', 'juv_fel_count', 
                'is_violent_recid', 'juv_misd_count', 'charge_degree']
gamma, alpha, test = audit.audit(feature_auditing, seed=3)
test_end = test[test.predicted == 1]

print(gamma)
print(alpha)
print(test_end[test_end.attr == 0].describe())
print(test_end[test_end.attr != 0].describe())
print((test_end[test_end.attr == 0].weight.sum() - \
         test_end[test_end.attr == 1].weight.sum()) / test_end.weight.sum())

results = audit.audit_iter(feature_auditing, nboot=100)
print(results.mean(axis=0))
print(np.sqrt(results.var(axis=0)))

"""
#DecisionTreeClassifier(max_depth=40, min_samples_leaf=0.01)
alpha, test_start, test_end = ad.get_violation_weight_iter(test, auditor, features_audit)
print(test_end[test_end.attr == 0][features_audit + ['predict']].describe())
print(test_end[test_end.attr == 1][features_audit + ['predict']].describe())
print(test_start[features_audit + ['predict']].describe())


# reduction method
epsilon = 0.01
constraint = moments.EO()
trainX = train[feature_list]
trainY = train[outcome]
trainA = train['attr'] 
logreg = LogisticRegression()
dct = DecisionTreeClassifier()
res_tuple = red.expgrad(trainX, trainA, trainY, dct,
							cons=constraint, eps=epsilon)
res = res_tuple._asdict()
best_classifier = res["best_classifier"]
test['predict'] = np.array(best_classifier(np.array(test[feature_list])))
test.loc[test.predict < 0.5, 'predict'] = 0
test.loc[test.predict > 0.5, 'predict'] = 1

# auditing learner
feature_audit = ['age_cat', 'priors_count', 'juv_fel_count', 'is_violent_recid']
score, _ = ad.audit_tree(test, feature_audit, 'predict', protected)
#print(unfair_treatment[unfair_treatment.sex == 'Caucasian'][feature_audit + ['predict', outcome]].describe())
#print(unfair_treatment[unfair_treatment.race == 'African-American'][feature_audit + ['predict', outcome]].describe())
print(score)
"""



