from aufair.simulations import repairs_method as rm
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from fairness.algorithms.baseline.LogisticRegression import LogisticRegression as LR
from fairness.algorithms.feldman import FeldmanAlgorithm as fa
from fairlearn import moments
from fairlearn import classred as red
from fairness.algorithms.baseline.SVM import SVM
from fairness.algorithms.feldman import FeldmanAlgorithm as fa
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

data_dict = {}

# data
train = pd.read_csv('C:\\Users\\xgitiaux\\Documents\\audit_fairness\\data\\adult_income_train_clean.csv')
test = pd.read_csv('C:\\Users\\xgitiaux\\Documents\\audit_fairness\\data\\adult_income_testclean.csv')
feature_list = ['age', 'workclass', 'education',  'occupation', 'gender',
		'hours-per-week', 'capital-gain', 'education-num', 'srace', 'marital-status']
features_auditing = ['age', 'workclass', 'education',  'occupation',
                    'hours-per-week', 'education-num']

data_dict['adult'] = {'train':train, 'test':test, 'features':feature_list, 'auditing_features': features_auditing}

# german credit
german = pd.read_csv('https://raw.githubusercontent.com/algofairness/fairness-comparison/master/fairness/data/preprocessed/german_numerical-binsensitive.csv')
german['Y'] = 2 * (german['credit'] == 1) - 1
german['attr'] = 2 * (german['sex'] == 1) - 1
features = [c for c in list(set(german.columns)) if c not in ['credit', 'attr', 'Y']]
features_auditing = [c for c in list(set(german.columns)) if c not in ['sex', 'credit', 'sex-age', 'age', 'attr', 'Y']]
#data_dict['german'] = {'train': german, 'test': german, 'features': features, 'auditing_features': features_auditing}
print(len(german))

# crimes data
crimes = pd.read_csv('C:\\Users\\xgitiaux\\Documents\\audit_fairness\\data\\communities_crime_clean.csv')
features = [c for c in list(set(crimes.columns)) if c not in ['ViolentCrimesPerPo',  'attr', 'Y']]
protected = pd.read_csv('https://raw.githubusercontent.com/sethneel/GerryFair/dev_branch/dataset/communities_protected.csv')
features_auditing = [c for c in features if crimes[c].values[0] == 0]
#data_dict['crimes'] = {'train':crimes, 'test':crimes, 'features':features, 'auditing_features': features_auditing}
print(len(crimes))

# auditing parameters
protected = ('attr', 1)
yname = 'outcome'
data = test.copy()
auditor = RandomForestClassifier(n_estimators=100, max_depth=2)
auditor = SVC(C=2)
report = pd.DataFrame()



# logistic regression
logreg = LR()
for dataname in data_dict.keys():
    train = data_dict[dataname]['train']
    data = data_dict[dataname]['test']
    feature_list = data_dict[dataname]['features']

    pred, _ = logreg.run(train[feature_list + ['attr', 'Y']], data[feature_list + ['attr', 'Y']], 'Y', 1, ['attr'],
                         'attr', 1, {'lambda': 1.0})
    data['outcome'] = pred
    features_auditing = data_dict[dataname]['auditing_features']
    audit = rm.audit(logreg, auditor, features_auditing, protected, yname)
    audit.standardize(data)
    violations, delta, delta_sd, tpp, tpp_sd = audit.get_violations(nboot=10)
    violations['experiment'] = 'lr'
    report.loc['baseline_logreg', 'delta_{}'.format(dataname)] = delta
    report.loc['baseline_logreg', 'delta_sd_{}'.format(dataname)] = delta_sd
    report.loc['baseline_logreg', 'tpp_{}'.format(dataname)] = tpp
    report.loc['baseline_logreg', 'tpp_sd_{}'.format(dataname)] = tpp_sd

    violations.to_csv(
        'C:\\Users\\xgitiaux\\Documents\\audit_fairness\\results\\groups_{}_logreg.csv'.format(dataname))


"""
svm = SVM()
for dataname in data_dict.keys():
    train = data_dict[dataname]['train']
    data = data_dict[dataname]['test']
    feature_list = data_dict[dataname]['features']

    pred, _ = svm.run(train[feature_list + ['attr', 'Y']], data[feature_list + ['attr', 'Y']], 'Y', 1, ['attr'],
                         'attr', 1, {'lambda': 1.0})
    data['outcome'] = pred
    features_auditing = data_dict[dataname]['auditing_features']
    audit = rm.audit(svm, auditor, features_auditing, protected, yname)
    audit.standardize(data)
    violations, delta, delta_sd, tpp, tpp_sd = audit.get_violations(nboot=10)
    violations['experiment'] = 'lr'
    report.loc['baseline_svm', 'delta_{}'.format(dataname)] = delta
    report.loc['baseline_svm', 'delta_sd_{}'.format(dataname)] = delta_sd
    report.loc['baseline_svm', 'tpp_{}'.format(dataname)] = tpp
    report.loc['baseline_svm', 'tpp_sd_{}'.format(dataname)] = tpp_sd


# experiment: feldman lr
feldman = fa.FeldmanAlgorithm(LR())
for dataname in data_dict.keys():
    print(dataname)
    train = data_dict[dataname]['train']
    data = data_dict[dataname]['test']
    feature_list = data_dict[dataname]['features']

    pred, _ = feldman.run(train[feature_list + ['attr', 'Y']], data[feature_list + ['attr', 'Y']], 'Y', 1, ['attr'],
                          'attr', 1, {'lambda': 1.0})
    data['outcome'] = pred
    features_auditing = data_dict[dataname]['auditing_features']
    audit = rm.audit(feldman, auditor, features_auditing, protected, yname)
    audit.standardize(data)
    violations, delta, delta_sd, tpp, tpp_sd = audit.get_violations(nboot=10)
    violations['experiment'] = 'fa_lr'
    report.loc['feldman_lr', 'delta_{}'.format(dataname)] = delta
    report.loc['feldman_lr', 'delta_sd_{}'.format(dataname)] = delta_sd
    report.loc['feldman_lr', 'tpp_{}'.format(dataname)] = tpp
    report.loc['feldman_lr', 'tpp_sd_{}'.format(dataname)] = tpp_sd

# feldman svm
feldman_svm = fa.FeldmanAlgorithm(SVM())
for dataname in data_dict.keys():
    train = data_dict[dataname]['train']
    data = data_dict[dataname]['test']
    feature_list = data_dict[dataname]['features']

    pred, _ = feldman_svm.run(train[feature_list + ['attr', 'Y']], data[feature_list + ['attr', 'Y']], 'Y', 1, ['attr'],
                              'attr', 1, {'lambda': 1.0})
    data['outcome'] = pred
    features_auditing = data_dict[dataname]['auditing_features']
    audit = rm.audit(feldman_svm, auditor, features_auditing, protected, yname)
    audit.standardize(data)
    violations, delta, delta_sd, tpp, tpp_sd = audit.get_violations(nboot=10)
    violations['experiment'] = 'fa_svm'
    report.loc['feldman_svm', 'delta_{}'.format(dataname)] = delta
    report.loc['feldman_svm', 'delta_sd_{}'.format(dataname)] = delta_sd
    report.loc['feldman_svm', 'tpp_{}'.format(dataname)] = tpp
    report.loc['feldman_svm', 'tpp_sd_{}'.format(dataname)] = tpp_sd
"""

# experiment fairlearn - LogReg
for dataname in data_dict.keys():
    logreg = LogisticRegression()
    train = data_dict[dataname]['train']
    data = data_dict[dataname]['test']
    feature_list = data_dict[dataname]['features']

    cons = moments.EO()
    epsilon = 0.05
    trainX = train[feature_list]
    trainY = train['Y']
    trainA = train['attr']

    res_tuple = red.expgrad(trainX, trainA, trainY, logreg, cons=cons, eps=epsilon)
    res = res_tuple._asdict()
    best_classifier = res["best_classifier"]
    logreg.fit(trainX, trainY.ravel())

    pred = best_classifier(np.array(data[feature_list]))
    pred[pred > 0.5] = 1
    pred[pred <= 0.5] = -1

    data['outcome'] = pred
    features_auditing = data_dict[dataname]['auditing_features']
    audit = rm.audit(logreg, auditor, features_auditing, protected, yname)
    audit.standardize(data)
    violations, delta, delta_sd, tpp, tpp_sd = audit.get_violations(nboot=10)
    violations['experiment'] = 'fairlearn'
    report.loc['fairlearn05', 'delta_{}'.format(dataname)] = delta
    report.loc['fairlearn05', 'delta_sd_{}'.format(dataname)] = delta_sd
    report.loc['fairlearn05', 'tpp_{}'.format(dataname)] = tpp
    report.loc['fairlearn05', 'tpp_sd_{}'.format(dataname)] = tpp_sd
    violations.to_csv('C:\\Users\\xgitiaux\\Documents\\audit_fairness\\results\\groups_{}_fairlearn2.csv'.format(dataname))

report.to_csv('C:\\Users\\xgitiaux\\Documents\\audit_fairness\\results\\certification_methods_all_test.csv')


