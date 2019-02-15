from fairness.algorithms.feldman import FeldmanAlgorithm as fa
from fairness.algorithms.zafar import ZafarAlgorithm as za
from fairness.algorithms.baseline.SVM import SVM
from fairness.algorithms.baseline.LogisticRegression import LogisticRegression as LR
import pandas as pd
import numpy as np
from aufair import auditing_data as ad
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from fairlearn import moments
from fairlearn import classred as red

# data: adult dataset
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
train['gender'] = train['sex'].astype('category').cat.codes
train['srace'] = train['race'].astype('category').cat.codes

test = pd.read_csv('..\\..\\..\\data\\adult_income_test.csv')

# clean features
test['workclass'] = test['workclass'].astype('category').cat.codes
test['education'] = test['education'].astype('category').cat.codes
test['occupation'] = test['occupation'].astype('category').cat.codes
test['relationship'] = test['relationship'].astype('category').cat.codes
test['marital-status'] = test['marital-status'].astype('category').cat.codes
test['income'] = test['income_bracket'].astype('category').cat.codes
test['gender'] = test['sex'].astype('category').cat.codes
test['srace'] = test['race'].astype('category').cat.codes

feature_list = ['age', 'workclass', 'education',  'occupation',
		'hours-per-week', 'education-num', 'srace', 'gender', 'marital-status']
outcome = 'income'
protected = {'sex': [' Male', ' Female'], 'race': [' Black', ' White']}

# features
train['income'] = 2 * (train['income_bracket'] == ' <=50K').astype('int32') - 1
test['income'] = 2 * (test['income_bracket'] == ' <=50K.').astype('int32') - 1
train['attr'] = 2 * (train.sex == ' Female').astype('int32') - 1
test['attr'] = 2 * (test.sex == ' Female').astype('int32') - 1
train = train[feature_list + ['attr', 'income']]
test = test[feature_list + ['attr', 'income']]

# define data to measure differential unfairness
features_auditing = ['age', 'workclass', 'education',  'occupation', 
                    'hours-per-week', 'education-num']
test_df = test.copy()

for var in features_auditing:
    test_df = test_df[~np.isnan(test_df[var])]
    test = test[~np.isnan(test[var])]
    test_df[var] = (test_df[var] - test_df[var].mean()) / np.sqrt(test_df[var].var())

# certifying lack of fairness
protected = ('attr', 1)
yname = 'outcome'
certification = pd.DataFrame()
certification.index.name = 'Experiment'

# experiment 1: baseline
logreg = LR()
pred, _ = logreg.run(train, test, 'income', 1, ['attr'], 'attr', 1, {'lambda': 1.0})
#logreg = LogisticRegression()
#logreg.fit(np.array(train[feature_list]), np.array(train[outcome].ravel()))
#pred = logreg.predict(np.array(test[feature_list]))

test_df['outcome'] = pred

auditor = RandomForestClassifier(n_estimators=100, max_depth=2)
audit = ad.detector_data(auditor, test_df, protected, yname, lw=0.005, niter=200, stepsize=0.01)
audit.get_y()

g, g_std = audit.certify_iter(features_auditing, yname,  nboot=1, balancing='MMD_NET')
certification.loc["Baseline_LogReg", "gamma"] = g
certification.loc["Baseline_LogReg", "gamma_deviation"] = g_std
certification.loc["Baseline_LogReg", "tpp"] = len(test_df[(test_df.attr == 1) & (test_df.outcome == 1) & (test_df.income == 1)]) / \
                                              len(test_df[(test_df.attr == 1) & (test_df.income == 1)])
certification.loc["Baseline_LogReg", "tpp"] = certification.loc["Baseline_LogReg", "tpp"] / \
                                              (len(test_df[(test_df.attr == -1) & (test_df.outcome == 1) & (test.income == 1)]) /\
                                              len(test_df[(test_df.attr == -1) & (test_df.income == 1)]))

# worst case violation
nboot = 1
resu = np.zeros(nboot)
for iter in range(nboot):
    delta, test_audited = audit.get_violation(features_auditing)
    resu[iter] = (test_audited[(test_audited.predicted == 1) & (test_audited.attr == 1)].outcome + 1).mean() / \
                 (test_audited[(test_audited.predicted == 1) & (test_audited.attr == -1)].outcome + 1).mean()
certification.loc["Baseline_LogReg", "delta"] = resu.mean()
print(test_audited[(test_audited.predicted == 1) & (test_audited.attr == 1)].describe())
print(test_audited[(test_audited.predicted == 1) & (test_audited.attr == -1)].describe())
print(certification)

# experiment 2: baseline - SVM
svm = SVM()
pred, _ = svm.run(train, test, 'income', 1, ['attr', 'srace'], 'attr', 1, {'lambda': 1.0})
test_df['outcome'] = pred

auditor = RandomForestClassifier(n_estimators=100, max_depth=2)
audit = ad.detector_data(auditor, test_df, protected, yname, lw=0.005, niter=200, stepsize=0.05)
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
certification.loc["Baseline_SVM", "delta"] = resu.mean()

# experiment 3: Feldman-Logreg
feldman = fa.FeldmanAlgorithm(LR())
pred, _ = feldman.run(train, test, 'income', 1, ['attr', 'srace'], 'attr', 1, {'lambda': 1.0})
test_df['outcome'] = pred

audit = ad.detector_data(auditor, test_df, protected, yname, lw=0.005, niter=200, stepsize=0.05)
audit.get_y()

g, g_std = audit.certify_iter(features_auditing, yname,  nboot=10, balancing='MMD_NET')
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
certification.loc["Feldman_LogReg", "delta"] = resu.mean()
print(certification)

# experiment 4: Feldman-SVM
feldman_svm = fa.FeldmanAlgorithm(SVM())
pred, _ = feldman_svm.run(train, test, 'income', 1, ['attr', 'srace'], 'attr', 1, {'lambda': 1.0})
test_df['outcome'] = pred

audit = ad.detector_data(auditor, test_df, protected, yname, lw=0.005, niter=200, stepsize=0.05)
audit.get_y()

g, g_std = audit.certify_iter(features_auditing, yname,  nboot=10, balancing='MMD_NET')
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
certification.loc["Feldman_SVM", "delta"] = resu.mean()

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

g, g_std = audit.certify_iter(features_auditing, yname,  nboot=10, balancing='MMD_NET')
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
certification.loc["Fairlearn_LR_05", "delta"] = resu.mean()

# experiment 6: fairlearn more restrictive ).01
cons = moments.EO()
res_tuple = red.expgrad(trainX, trainA, trainY, logreg, cons=cons, eps=0.1)
res = res_tuple._asdict()
best_classifier = res["best_classifier"]
logreg.fit(trainX, trainY.ravel())

pred = best_classifier(np.array(test[['age', 'workclass', 'education',  'occupation', 'gender'
		'hours-per-week', 'capital-gain', 'education-num', 'srace', 'marital-status']]))
pred[pred > 0.5] = 1
pred[pred <= 0.5] = -1
test_df['outcome'] = pred
print(len(test_df[(test_df.attr == 1) & (test_df.outcome == 1) & (test_df.income ==1)]) /
len(test_df[(test_df.attr == 1) & (test_df.income ==1)]))
print(len(test_df[(test_df.attr == -1) & (test_df.outcome == 1) & (test_df.income ==1)]) /
len(test_df[(test_df.attr == -1) & (test_df.income ==1)]))

audit = ad.detector_data(auditor, test_df, protected, yname, lw=0.005, niter=200, stepsize=0.05)
audit.get_y()

g, g_std = audit.certify_iter(features_auditing, yname,  nboot=10, balancing='MMD_NET')
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
certification.loc["Fairlearn_LR_1", "delta"] = resu.mean()

certification.to_csv('../../../results/certification_methods_adults.csv')

""""
# experiment 5: zafar algorithm
zafar = za.ZafarAlgorithmFairness()
pred, _ = zafar.run(train, test, 'income', 1, ['attr', 'srace'], 'attr', 1, {'c': 0.001})
test_df['outcome'] = pred

audit = ad.detector_data(auditor, test_df, protected, yname, lw=0.005, niter=200, stepsize=0.05)
audit.get_y()

g, g_std = audit.certify_iter(features_auditing, yname,  nboot=10, balancing='MMD_NET')
certification.loc["ZafarFairness", "gamma"] = g
certification.loc["ZafarFairness", "gamma_deviation"] = g_std

"""
print(certification)



