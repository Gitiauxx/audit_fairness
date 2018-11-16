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
    data['noise'] = np.random.normal(scale=0.0, size=N)
    data['y'] =  data['x1'] + data['x2'] + data['noise'] 
    data['outcome'] = (data.y >= 0).astype('int32')

    # split the data into train versus test set using a 0.7/0.3 ratio
    train = data.loc[np.random.choice(data.index, int(len(data)* 0.7), replace=True), :]
    test = data.drop(train.index)

    # train data is used to fit the audited learner
    train_x = np.array(train[['x1', 'x2']])
    train_y = np.array(train['outcome'].ravel())
    audited = LogisticRegression()
    audited.fit(train_x, train_y)

    # test data is used to simulate oracle
    protected = {'attr': [0, 1]}
    features = ['x1', 'x2']
    test = ao.oracle_access(test, features, audited, protected)

    # here is where unfairness is created 
    results = pd.DataFrame()
    for gamma in np.arange(nu_max):
        gamma = gamma / 10 

        # copy test
        test_sim = test.copy()
        ind = np.random.choice(test_sim.index, int(gamma * len(test)), replace=False)
        #test_sim.loc[ind, 'predict_attr_0'] = 1
        # - test_sim['predict_attr_0']
        test_sim.loc[ind, 'predict_attr_1'] = 1 - test_sim.loc[ind, 'predict_attr_0']
    
        # auditing using a decision tree
        auditor = DecisionTreeClassifier()
        res = ao.binary_audit(test_sim, features, auditor, protected, seed=1)
        results.loc[gamma,  'unfairness'] = res['attr']
    
    return results

if __name__ == "__main__":
    N = 300000
    nu_max = 10
    results = test_experiment1(N, nu_max)
    print(results)


