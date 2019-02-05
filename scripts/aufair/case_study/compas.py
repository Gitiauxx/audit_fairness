import pandas as pd
import numpy as np
from aufair import auditing_data as ad

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# load data
data = pd.read_csv('..\\..\\..\\data\\compas-scores-two-years.csv')
print(data.columns)

# create categorical data for age_cat, sex, race and charge degree
data['gender'] = data.sex.astype('category').cat.codes
data['age_cat'] = data.age_cat.astype('category').cat.codes
data['charge_degree'] = data.c_charge_degree.astype('category').cat.codes
data['crace'] = data.race.astype('category').cat.codes
data['is_violent_recid'] = data.is_violent_recid.astype('category').cat.codes
data['juv_fel_count'] = data.juv_fel_count.astype('category').cat.codes
data['is_recid'] = data.is_recid.astype('category').cat.codes
data['count_race']  = data['priors_count'] * data['crace']
data['outcome'] = 2 * (data.v_score_text.isin(['High', 'Medium'])).astype('int32') - 1

feature_list = ['age_cat',  'priors_count', 'charge_degree', 'juv_fel_count', 'is_recid']
for var in feature_list:
    data = data[~np.isnan(data[var])]
    data[var + '_s'] = (data[var] - data[var].mean()) / data[var].var() 

# protected attribute: race
data['attr'] = 2 * (data['race'] == 'African-American').astype('int32') - 1
print(data.groupby(['attr', 'outcome']).size())
outcome = 'outcome'
protected = ('attr', 1)
yname = 'outcome'

auditor = DecisionTreeClassifier(max_depth= 5)
#auditor = SVC()
auditor = RandomForestClassifier(n_estimators=20, max_depth=2)
features_auditing = ['age_cat_s',  'priors_count_s', 'charge_degree_s', 'juv_fel_count_s', 'is_recid_s']

audit = ad.detector_data(auditor, data, protected, yname, lw=0.005, niter=50, stepsize=0.05)
audit.get_y()
g, g_std = audit.certify_iter(features_auditing, yname,  nboot=1, balancing='MMD_NET')
print(g)
print(g_std)

# identify worst violation
gamma, test = audit.get_violation(feature_list)
print(test[(test.predicted == 1) & (test.attr == 1)][feature_list + ['outcome', 'attr']].describe())
print(test[(test.predicted == 1) & (test.attr == -1)][feature_list + ['outcome', 'attr']].describe())
print(gamma)

# individual 
bob = np.array([1.0, 1, 1, 0, 0]).ravel()

bob[0] = (bob[0] - data.age_cat.mean()) / data.age_cat.var()
bob[1] = (bob[1] - data.priors_count.mean()) / data.priors_count.var()
bob[2] = (bob[2] - data.charge_degree.mean()) / data.charge_degree.var()
bob[3] = (bob[3] - data.juv_fel_count.mean()) / data.juv_fel_count.var()
bob[4] = (bob[4] - data.is_recid.mean()) / data.is_recid.var()

gamma = audit.get_violation_individual(feature_list, bob)
print(gamma)

andy = np.array([8.0, 1, 1, 2, 0]).ravel()

andy[0] = (andy[0] - data.age_cat.mean()) / data.age_cat.var()
andy[1] = (andy[1] - data.priors_count.mean()) / data.priors_count.var()
andy[2] = (andy[2] - data.charge_degree.mean()) / data.charge_degree.var()
andy[3] = (andy[3] - data.juv_fel_count.mean()) / data.juv_fel_count.var()
andy[4] = (andy[4] - data.is_recid.mean()) / data.is_recid.var()

gamma = audit.get_violation_individual(feature_list, andy)
print(gamma)