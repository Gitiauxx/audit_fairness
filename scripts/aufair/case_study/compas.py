import pandas as pd
import numpy as np
from aufair import auditing_data as ad

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# load data
data = pd.read_csv('..\\..\\..\\data\\compas-scores-two-years.csv')

# create categorical data for age_cat, sex, race and charge degree
data['gender'] = data.sex.astype('category').cat.codes
data['age_cat'] = data.age_cat.astype('category').cat.codes
data['charge_degree'] = data.c_charge_degree.astype('category').cat.codes
data['crace'] = data.race.astype('category').cat.codes
data['is_violent_recid'] = data.is_violent_recid.astype('category').cat.codes
data['juv_fel_count'] = data.juv_fel_count.astype('category').cat.codes
data['is_recid'] = data.is_recid.astype('category').cat.codes
data['count_race'] = data['priors_count'] * data['crace']
data['outcome'] = 2 * (data.v_score_text.isin(['High'])).astype('int32') - 1
data['high_risk'] = (data.outcome == 1).astype('int32')

feature_list = ['age_cat',  'priors_count', 'charge_degree', 'juv_fel_count', 'juv_misd_count']
for var in feature_list:
    data = data[~np.isnan(data[var])]
    data[var + '_s'] = (data[var] - data[var].mean()) / data[var].var() ** 0.5

# export data for summary statistics
data.to_csv('..\\..\\..\\data\\compas-scores-cleaned.csv')

# protected attribute: race
data['attr'] = 2 * (data['race'] == 'African-American').astype('int32') - 1
outcome = 'outcome'
protected = ('attr', 1)
yname = 'outcome'

# certify lack of differential fairness
certification = pd.DataFrame()
certification.index.name = 'Experiment'

auditor = RandomForestClassifier(n_estimators=100, max_depth=2)
#auditor = SVC()
audit = ad.detector_data(auditor, data, protected, yname, lw=0.005, niter=200, stepsize=0.05)
audit.get_y()

"""
# experiment 1
features_auditing = ['age_cat_s',  'priors_count_s', 'charge_degree_s', 'juv_fel_count_s', 'juv_misd_count_s']
g, g_std = audit.certify_iter(features_auditing, yname,  nboot=100, balancing='MMD_NET')
certification.loc["1", "gamma"] = g
certification.loc["1", "gamma_deviation"] = g_std

features_auditing = ['age_cat_s',  'priors_count_s', 'charge_degree_s']
g, g_std = audit.certify_iter(features_auditing, yname,  nboot=100, balancing='MMD_NET')
certification.loc["2", "gamma"] = g
certification.loc["2", "gamma_deviation"] = g_std

features_auditing = ['priors_count_s', 'charge_degree_s']
g, g_std = audit.certify_iter(features_auditing, yname,  nboot=100, balancing='MMD_NET')
certification.loc["3", "gamma"] = g
certification.loc["3", "gamma_deviation"] = g_std

certification.to_csv('../../../results/compas_certification_gender.csv')


# identify worst violation -- iter
nboot = 100
features_auditing = ['age_cat_s',  'priors_count_s', 'charge_degree_s', 'juv_fel_count_s', 'juv_misd_count_s']
results = pd.DataFrame()
variables = ['priors_count', 'charge_degree', 'juv_fel_count', 'juv_misd_count', 'high_risk']
for iter in range(nboot):
    delta, test = audit.get_violation(features_auditing)
    for v in variables:
        results.loc[iter, v + '_1'] = test[(test.predicted == 1) & (test.attr == 1)][v].mean()
        results.loc[iter, v + '_0'] = test[(test.predicted == 1) & (test.attr == -1)][v].mean()
    results['attr_1'] = len(test[(test.predicted == 1) & (test.attr == 1)])
    results['attr_0'] = len(test[(test.predicted == 1) & (test.attr == -1)])
    results.loc[iter, 'delta'] = delta
results.to_csv('../../../results/compas_most_harm.csv')
"""

# individual 
bob = np.array([2.0, 1, 1, 0, 0]).ravel()

bob[0] = (bob[0] - data.age_cat.mean()) / data.age_cat.var() ** 0.5
bob[1] = (bob[1] - data.priors_count.mean()) / data.priors_count.var() ** 0.5
bob[2] = (bob[2] - data.charge_degree.mean()) / data.charge_degree.var() ** 0.5
bob[3] = (bob[3] - data.juv_fel_count.mean()) / data.juv_fel_count.var() ** 0.5
bob[4] = (bob[4] - data.juv_misd_count.mean()) / data.juv_misd_count.var() ** 0.5


gamma = audit.get_violation_individual(feature_list, bob)
print(gamma)


andy = np.array([2, 4, 1, 0, 0]).ravel()

andy[0] = (andy[0] - data.age_cat.mean()) / data.age_cat.var() ** 0.5
andy[1] = (andy[1] - data.priors_count.mean()) / data.priors_count.var() ** 0.5
andy[2] = (andy[2] - data.charge_degree.mean()) / data.charge_degree.var() ** 0.5
andy[3] = (andy[3] - data.juv_fel_count.mean()) / data.juv_fel_count.var() ** 0.5
andy[4] = (andy[4] - data.juv_misd_count.mean()) / data.juv_misd_count.var() ** 0.5

gamma1 = audit.get_violation_individual(feature_list, andy)
print(gamma1)

