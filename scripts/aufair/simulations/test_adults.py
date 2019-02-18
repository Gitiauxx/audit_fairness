import pandas as pd
import numpy as np
from aufair import auditing_data as ad
from fairness.algorithms.baseline.SVM import SVM

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from fairness.algorithms.baseline.LogisticRegression import LogisticRegression as LR
from fairness.algorithms.feldman import FeldmanAlgorithm as fa
from fairlearn import moments
from fairlearn import classred as red

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

# define outcome and attribute
train['income'] = 2 * (train['income_bracket'] == ' <=50K').astype('int32') - 1
test['income'] = 2 * (test['income_bracket'] == ' <=50K.').astype('int32') - 1
train['attr'] = 2 * (train.sex == ' Female').astype('int32') - 1
test['attr'] = 2 * (test.sex == ' Female').astype('int32') - 1

feature_list = ['age', 'workclass', 'education',  'occupation', 
		'hours-per-week', 'education-num', 'srace', 'gender', 'marital-status']
outcome = 'income'
protected = {'sex': [' Male', ' Female'], 'race':[' Black', ' White']}

# auditing configuration
features_auditing = ['age', 'workclass', 'education',  'occupation',
                    'hours-per-week', 'education-num']
test_df = test.copy()

for var in features_auditing:
    test_df = test_df[~np.isnan(test_df[var])]
    test = test[~np.isnan(test[var])]
    test_df[var  +'_s'] = (test_df[var] - test_df[var].mean()) / np.sqrt(test_df[var].var())
features_standardized = [v+ '_s' for v in features_auditing]

protected = ('attr', 1)
yname = 'outcome'
certification = pd.DataFrame()
certification.index.name = 'Experiment'

# logistic regression
logreg = LR()
pred, _ = logreg.run(train[feature_list + ['attr', 'income']], test[feature_list + ['attr', 'income']], 'income', 1, ['attr'], 'attr', 1, {'lambda': 1.0})
test_df['outcome'] = pred

# auditing the classifier
auditor = RandomForestClassifier(n_estimators=100, max_depth=2)
audit = ad.detector_data(auditor, test_df, protected, yname, lw=0.005, niter=100, stepsize=0.01)
audit.get_y()
g, g_std = audit.certify_iter(features_auditing, yname,  nboot=1, balancing='MMD_NET')
certification.loc["Baseline_LogReg", "gamma"] = g
certification.loc["Baseline_LogReg", "gamma_deviation"] = g_std
certification.loc["Baseline_LogReg", "tpp"] = len(test_df[(test_df.attr == 1) & (test_df.outcome == 1) & (test_df.income == 1)]) / \
                                              len(test_df[(test_df.attr == 1) & (test_df.income == 1)])
certification.loc["Baseline_LogReg", "tpp"] = certification.loc["Baseline_LogReg", "tpp"] / \
                                              (len(test_df[(test_df.attr == -1) & (test_df.outcome == 1) & (test.income == 1)]) /\
                                              len(test_df[(test_df.attr == -1) & (test_df.income == 1)]))

nboot = 10
resu = np.zeros(nboot)
for iter in range(nboot):
    delta, test_audited = audit.get_violation(features_auditing)
    resu[iter] = (test_audited[(test_audited.predicted == 1) & (test_audited.attr == 1)].outcome + 1).mean() / \
                 (test_audited[(test_audited.predicted == 1) & (test_audited.attr == -1)].outcome + 1).mean()

certification.loc["Baseline_LogReg", "delta"] = resu[~np.isnan(resu)].mean()

print(test_audited[(test_audited.predicted == 1) & (test_audited.attr == 1)][features_auditing + ['outcome']] .describe())
print(test_audited[(test_audited.predicted == 1) & (test_audited.attr == -1)][features_auditing + ['outcome']].describe())

# experiment 2: baseline - SVM
svm = SVM()
pred, _ = svm.run(train[feature_list + ['attr', 'income']], test[feature_list + ['attr', 'income']], 'income', 1, ['attr'], 'attr', 1, {'lambda': 1.0})
test_df['outcome'] = pred

auditor = RandomForestClassifier(n_estimators=100, max_depth=2)
audit = ad.detector_data(auditor, test_df, protected, yname, lw=0.01, niter=100, stepsize=0.01)
audit.get_y()

g, g_std = audit.certify_iter(features_auditing, yname,  nboot=10, balancing='MMD_NET')
certification.loc["Baseline_SVM", "gamma"] = g
certification.loc["Baseline_SVM", "gamma_deviation"] = g_std
certification.loc["Baseline_SVM", "tpp"] = len(test_df[(test_df.attr == 1) & (test_df.outcome == 1) & (test_df.income == 1)]) / \
                                              len(test_df[(test_df.attr == 1) & (test_df.income == 1)])
certification.loc["Baseline_SVM", "tpp"] = certification.loc["Baseline_SVM", "tpp"] / \
                                              (len(test_df[(test_df.attr == -1) & (test_df.outcome == 1) & (test.income == 1)]) /\
                                              len(test_df[(test_df.attr == -1) & (test_df.income == 1)]))


for iter in range(nboot):
    delta, test_audited = audit.get_violation(features_auditing)
    resu[iter] = (test_audited[(test_audited.predicted == 1) & (test_audited.attr == 1)].outcome + 1).mean() / \
                 (test_audited[(test_audited.predicted == 1) & (test_audited.attr == -1)].outcome + 1).mean()
certification.loc["Baseline_SVM", "delta"] = resu[~np.isnan(resu)].mean()

# experiment 3: Feldman-Logreg
feldman = fa.FeldmanAlgorithm(LR())
pred, _ = feldman.run(train[feature_list + ['attr', 'income']], test[feature_list + ['attr', 'income']], 'income', 1, ['attr'], 'attr', 1, {'lambda': 1.0})
test_df['outcome'] = pred

audit = ad.detector_data(auditor, test_df, protected, yname, lw=0.005, niter=200, stepsize=0.05)
audit.get_y()

g, g_std = audit.certify_iter(features_auditing, yname,  nboot=1, balancing='MMD_NET')
certification.loc["Feldman_LogReg", "gamma"] = g
certification.loc["Feldman_LogReg", "gamma_deviation"] = g_std
certification.loc["Feldman_LogReg", "tpp"] = len(test_df[(test_df.attr == 1) & (test_df.outcome == 1) & (test_df.income == 1)]) / \
                                              len(test_df[(test_df.attr == 1) & (test_df.income == 1)])
certification.loc["Feldman_LogReg", "tpp"] = certification.loc["Feldman_LogReg", "tpp"] / \
                                              (len(test_df[(test_df.attr == -1) & (test_df.outcome == 1) & (test.income == 1)]) /\
                                              len(test_df[(test_df.attr == -1) & (test_df.income == 1)]))

for iter in range(nboot):
    delta, test_audited = audit.get_violation(features_auditing)
    resu[iter] = (test_audited[(test_audited.predicted == 1) & (test_audited.attr == 1)].outcome + 1).mean() / \
                 (test_audited[(test_audited.predicted == 1) & (test_audited.attr == -1)].outcome + 1).mean()
certification.loc["Feldman_LogReg", "delta"] = resu[~np.isnan(resu)].mean()

# experiment 4: Feldman-SVM
feldman_svm = fa.FeldmanAlgorithm(SVM())
pred, _ = feldman_svm.run(train[feature_list + ['attr', 'income']], test[feature_list + ['attr', 'income']], 'income', 1, ['attr'], 'attr', 1, {'lambda': 1.0})
test_df['outcome'] = pred

audit = ad.detector_data(auditor, test_df, protected, yname, lw=0.005, niter=200, stepsize=0.05)
audit.get_y()

g, g_std = audit.certify_iter(features_auditing, yname,  nboot=1, balancing='MMD_NET')
certification.loc["Feldman_SVM", "gamma"] = g
certification.loc["Feldman_SVM", "gamma_deviation"] = g_std
certification.loc["Feldman_SVM", "tpp"] = len(test_df[(test_df.attr == 1) & (test_df.outcome == 1) & (test_df.income == 1)]) / \
                                              len(test_df[(test_df.attr == 1) & (test_df.income == 1)])
certification.loc["Feldman_SVM", "tpp"] = certification.loc["Feldman_SVM", "tpp"] / \
                                              (len(test_df[(test_df.attr == -1) & (test_df.outcome == 1) & (test.income == 1)]) /\
                                              len(test_df[(test_df.attr == -1) & (test_df.income == 1)]))

for iter in range(nboot):
    delta, test_audited = audit.get_violation(features_auditing)
    resu[iter] = (test_audited[(test_audited.predicted == 1) & (test_audited.attr == 1)].outcome + 1).mean() / \
                 (test_audited[(test_audited.predicted == 1) & (test_audited.attr == -1)].outcome + 1).mean()
certification.loc["Feldman_SVM", "delta"] = resu[~np.isnan(resu)].mean()
print(certification)

# experiment 5 - fairlearn - LogReg
logreg = LogisticRegression()
cons = moments.EO()
epsilon = 0.02
trainX = train[['age', 'workclass', 'education',  'occupation', 'gender',
		'hours-per-week', 'capital-gain', 'education-num', 'srace', 'marital-status']]
trainY = train['income']
trainA = train['attr']

res_tuple = red.expgrad(trainX, trainA, trainY, logreg, cons=cons, eps=0.05)
res = res_tuple._asdict()
best_classifier = res["best_classifier"]
logreg.fit(trainX, trainY.ravel())

pred = best_classifier(np.array(test[['age', 'workclass', 'education',  'occupation', 'gender',
		'hours-per-week', 'capital-gain', 'education-num', 'srace', 'marital-status']]))
pred[pred > 0.5] = 1
pred[pred <= 0.5] = -1
test_df['outcome'] = pred

audit = ad.detector_data(auditor, test_df, protected, yname, lw=0.005, niter=200, stepsize=0.1)
audit.get_y()

g, g_std = audit.certify_iter(features_auditing, yname,  nboot=1, balancing='MMD_NET')
certification.loc["Fairlearn_LR_05", "gamma"] = g
certification.loc["Fairlearn_LR_05", "gamma_deviation"] = g_std
certification.loc["Fairlearn_LR_05", "tpp"] = len(test_df[(test_df.attr == 1) & (test_df.outcome == 1) & (test_df.income == 1)]) / \
                                              len(test_df[(test_df.attr == 1) & (test_df.income == 1)])
certification.loc["Fairlearn_LR_05", "tpp"] = certification.loc["Fairlearn_LR_05", "tpp"] / \
                                              (len(test_df[(test_df.attr == -1) & (test_df.outcome == 1) & (test.income == 1)]) /\
                                              len(test_df[(test_df.attr == -1) & (test_df.income == 1)]))

for iter in range(nboot):
    delta, test_audited = audit.get_violation(features_auditing)
    resu[iter] = (test_audited[(test_audited.predicted == 1) & (test_audited.attr == 1)].outcome + 1).mean() / \
                 (test_audited[(test_audited.predicted == 1) & (test_audited.attr == -1)].outcome + 1).mean()
certification.loc["Fairlearn_LR_05", "delta"] = resu[~np.isnan(resu)].mean()



# experiment 6: fairlearn more restrictive ).01
cons = moments.EO()
res_tuple = red.expgrad(trainX, trainA, trainY, logreg, cons=cons, eps=0.1)
res = res_tuple._asdict()
best_classifier = res["best_classifier"]
logreg.fit(trainX, trainY.ravel())

pred = best_classifier(np.array(test[['age', 'workclass', 'education',  'occupation', 'gender',
		'hours-per-week', 'capital-gain', 'education-num', 'srace', 'marital-status']]))
pred[pred > 0.5] = 1
pred[pred <= 0.5] = -1
test_df['outcome'] = pred


audit = ad.detector_data(auditor, test_df, protected, yname, lw=0.005, niter=200, stepsize=0.05)
audit.get_y()

g, g_std = audit.certify_iter(features_auditing, yname,  nboot=1, balancing='MMD_NET')
certification.loc["Fairlearn_LR_1", "gamma"] = g
certification.loc["Fairlearn_LR_1", "gamma_deviation"] = g_std
certification.loc["Fairlearn_LR_1", "tpp"] = len(test_df[(test_df.attr == 1) & (test_df.outcome == 1) & (test_df.income == 1)]) / \
                                              len(test_df[(test_df.attr == 1) & (test_df.income == 1)])
certification.loc["Fairlearn_LR_1", "tpp"] = certification.loc["Fairlearn_LR_1", "tpp"] / \
                                              (len(test_df[(test_df.attr == -1) & (test_df.outcome == 1) & (test.income == 1)]) /\
                                              len(test_df[(test_df.attr == -1) & (test_df.income == 1)]))

for iter in range(nboot):
    delta, test_audited = audit.get_violation(features_auditing)
    resu[iter] = (test_audited[(test_audited.predicted == 1) & (test_audited.attr == 1)].outcome + 1).mean() / \
                 (test_audited[(test_audited.predicted == 1) & (test_audited.attr == -1)].outcome + 1).mean()
certification.loc["Fairlearn_LR_1", "delta"] = resu[~np.isnan(resu)].mean()


certification.to_csv('../../../results/certification_methods_adults2.csv')

print(certification)


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
        
