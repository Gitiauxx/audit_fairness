from aufair import audit_oracle as ao
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


def test_experiment1(N, nu_max):
    """
    Simulate a bivariate linear classification predicted by 
    a logistic regression. Noise is added using a gaussian process
    Then audit using oracle access using 
    protected = [0, 1] as a protected attributes

    The level of unfairness from the classifier is forced 
    by randomly selected some data points and set their label
    to 0 if protected = 0 and 1 if protected = 1.
 
    Parameters
    ------------
    N: integer
        Size of the overall data 
    nu_max: integer
        maximum amount if unfairness to inject in the experiment

    """

    # simulate a synthethic data
    data = pd.DataFrame(index=np.arange(N))
    data['attr'] = np.random.choice([0, 1], N)
    data['x1'] = np.random.normal(size=N) 
    data['x2'] = np.random.normal(size=N)
    data['noise'] = np.random.normal(scale=0.5, size=N)
    data['y'] =  data['x1'] + data['x2'] + data['noise'] 
    data['outcome'] = (data.y >= 0).astype('int32')

    # split the data into train versus test set using a 0.7/0.3 ratio
    train = data.loc[np.random.choice(data.index, int(len(data)* 0.7), replace=True), :]
    test = data.drop(train.index)

    # train data is used to fit the audited learner
    features = ['x1', 'x2']
    train_x = np.array(train[features])
    train_y = np.array(train['outcome'].ravel())
    audited = LogisticRegression()
    audited.fit(train_x, train_y)

    # test data is used to simulate oracle
    protected = {'attr': [0, 1]}
    test = ao.oracle_access(test, features, audited, protected)

    # here is where unfairness is created 
    results = pd.DataFrame()
    results.index.name = 'gamma'
    
    for gamma in np.arange(nu_max):
        gamma = gamma / 10 

        # copy test to avoid changes to be compounded
        test_sim = test.copy()
        ind = np.random.choice(test_sim.index, int(gamma * len(test)), replace=False)
        test_sim.loc[ind, 'predict_attr_1'] = 1 - test_sim.loc[ind, 'predict_attr_0']
    
        # auditing using a decision tree with x1 and x2
        auditor = DecisionTreeClassifier()
        features_audit = ['x1', 'x2']
        res = ao.binary_audit(test_sim, features_audit, auditor, protected, seed=1)
        results.loc[gamma,  'unfairness_all'] = res['attr']

        # auditing using a decision tree with x1 only
        features_audit = ['x1']
        res = ao.binary_audit(test_sim, features_audit, auditor, protected, seed=1)
        results.loc[gamma,  'unfairness_x1'] = res['attr']
    
    return results


def test_experiment2(N):
    """
    Simulate a bivariate linear classification predicted by 
    a logistic regression. Noise is added using a gaussian process
    Then audit using oracle access using 
    protected = [-1, 1] as a protected attributes

    The level of unfairness from the classifier is forced 
    by adding protected variables in the audited learner
 
    Parameters
    ------------
    N: integer
        Size of the overall data 
    nu_max: integer
        maximum amount if unfairness to inject in the experiment

    """

    # simulate a synthethic data
    data = pd.DataFrame(index=np.arange(N))
    data['attr'] = np.random.choice([-1, 1], N)
    data['x1'] = np.random.normal(size=N) 
    data['x2'] = np.random.normal(size=N)
    data['noise'] = np.random.normal(scale=0.5, size=N)
    data['y'] =  data['x1'] + data['x2'] + data['attr'] + data['noise']
    data['outcome'] = (data.y >= 0).astype('int32')

    # split the data into train versus test set using a 0.7/0.3 ratio
    train = data.loc[np.random.choice(data.index, int(len(data)* 0.7), replace=True), :]
    test = data.drop(train.index)

    # train data is used to fit the audited learner
    features = ['x1', 'x2', 'attr']
    train_x = np.array(train[features])
    train_y = np.array(train['outcome'].ravel())
    audited = LogisticRegression()
    audited.fit(train_x, train_y)

    # test data is used to simulate oracle
    protected = {'attr': [-1, 1]}
    test = ao.oracle_access(test, features, audited, protected)

    # here is where unfairness is created 
    results = pd.DataFrame()
    
    # auditing with x1 and logistic classification
    features_audit = ['x1']
    auditor = LogisticRegression()
    res = ao.binary_audit(test, features_audit, auditor, protected, seed=1)
    results.loc['x1',  'unfairness'] = res['attr']

    # auditing with x1 and x2 and logistic classification
    features_audit = ['x1', 'x2', 'x1x2', 'x1x22']
    test['x1x2'] = test['x1'] * test['x2']
    test['x1x22'] = test['x2'] ** 2
    auditor = LogisticRegression()
    res = ao.binary_audit(test, features_audit, auditor, protected, seed=1)
    results.loc['x1 + x2',  'unfairness'] = res['attr']

    # auditing with x1 and decision tree
    features_audit = ['x1']
    auditor = DecisionTreeClassifier()
    res = ao.binary_audit(test, features_audit, auditor, protected, seed=1)
    results.loc['x1 - dct',  'unfairness'] = res['attr']

    # auditing with x1 and x2 and decision tree
    features_audit = ['x1', 'x2']
    auditor = DecisionTreeClassifier()
    res = ao.binary_audit(test, features_audit, auditor, protected, seed=1)
    results.loc['x1 + x2 - dct',  'unfairness'] = res['attr']
    
    return results

if __name__ == "__main__":
    N = 10000
    nu_max = 10
    results = test_experiment1(N, nu_max)
    print(results)
    #results.to_csv("../../results/synth_oracle_exp1.csv")


