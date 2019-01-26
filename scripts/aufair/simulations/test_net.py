from aufair import auditing_net as an
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Ridge, BayesianRidge, LassoLars
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

def test_certifying(n, n_test, nu_min, nu_max, sigma_noise=0.2, unbalance=0.0, nboot=None, balancing=None):
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
    data = pd.DataFrame(index=np.arange(n))

    data['x1'] = np.random.normal(size=n) 
    data['x2'] = np.random.normal(size=n)
    data['x3'] = np.random.normal(size=n) + data['x1']
    data['x4'] = np.random.normal(size=n) * data['x2']
    data['noise'] = np.random.normal(scale=0.2, size=n)
    data['w'] =  np.exp(unbalance * data['x2']  - unbalance * data['x1']  + data['noise'] )
    data['w'] = data['w'] / (1 + data['w'])
    data['u'] = np.random.uniform(0, 1, size=len(data))
    data.loc[data.u < data.w, 'attr'] = 1
    data.loc[data.u >= data.w, 'attr'] = -1
    
    data['noise'] = np.random.normal(scale=sigma_noise, size=n)
    data['y'] =  data['x2'] + data['x1'] - data['x4'] + data['x3']
                #data['x1'] * data['x3'] - data['x2'] * data['x3'] + data['noise']  
    data['outcome'] = - 1 + 2 * (data.y >= 0).astype('int32')

    # split the data into train versus test set using a 0.7/0.3 ratio
    train = data.loc[np.random.choice(data.index, int(len(data)* 0.7), replace=True), :]
    test = data.drop(train.index)
    test = test.loc[np.random.choice(test.index, n_test, replace=False), :]

    # train data is used to fit the audited learner
    features = ['x1', 'x2', 'x3', 'x4']
    train_x = np.array(train[features])
    train_y = np.array(train['outcome'].ravel())
    audited = LogisticRegression()
    audited.fit(train_x, train_y)

    # test data is used to audit for fairness
    test = data
    protected = {'attr': 1}
    test_x = np.array(data[features])
    test['predict'] = audited.predict(test_x)

    # here is where unfairness is created 
    results = pd.DataFrame()
    results.index.name = 'nu'

    for gamma in np.arange(nu_min, nu_max):
        gamma = gamma / 10 
        print(gamma)
      
        # copy test to avoid changes to be compounded
        test_sim = test.copy()
        test_sim.set_index(np.arange(len(test_sim)))
        
        # define violation shape
        mask = (test_sim.x1**2 + test_sim.x2**2 <= 1) & (test_sim.predict == -1)
        #mask = (test_sim.x1 <= -0) & (test_sim.x2 <= -0)
        mask1 = ((test_sim.attr == 1)  & (mask))
        mask2 = ((test_sim.attr == -1) & (mask))
        
        np.random.seed(seed=1)
        l = len(test_sim[mask1])
        ind = np.random.choice(test_sim[mask1].index, 
                                int(l), 
                                replace=False)

        test_sim.loc[ind, 'predict'] = (-1) * test_sim.loc[ind, 'predict']
        ll = len(test_sim[(mask1) & (test_sim.predict == 1)])

        l2 = len(test_sim[mask2])
        ind = np.random.choice(test_sim[mask2].index, 
                                int((1-gamma) * l2), 
                                replace=False)

        test_sim.loc[ind, 'predict'] = (-1) * test_sim.loc[ind, 'predict']
        ll2 = len(test_sim[(mask2) & (test_sim.predict == 1)])

        # auditing using a decision tree with x1 and x2
        protected = ('attr', 1)
        yname = 'predict'
        audit = an.audit_net(test_sim, protected, yname, 4)
        audit.get_y()

        feature_auditing = ['x1', 'x2', 'x3', 'x4']
        g, g_std = audit.certify_iter(features, 'predict', balancing=balancing, nboot=nboot)

        if gamma < 1:
            delta = ll / (ll + ll2) 
            print(delta)
            delta = delta / (1 - delta)
            delta = delta / l * l2
            
            delta = (delta - 1) / (2 * (delta + 1)) * (ll + ll2) / len(test_sim)
        else:
            delta = 1/2 * (ll + ll2) / len(test_sim)
        
        results.loc[gamma, 'gamma'] = delta 
        results.loc[gamma,  'estimated_gamma'] = g
        results.loc[gamma, 'bias'] = g  / results.loc[gamma, 'gamma']
    
    return results


if __name__ == "__main__":
    n = 500000
    n_test = 10000
    nu_max = 3
    nu_min = 1
    unbalance = 0.2
    
    results = test_certifying(n, n_test, nu_min, nu_max, nboot=20, unbalance=unbalance, balancing="MMD")
    print(results)
    #results.to_csv("../../results/synth_oracle_exp1.csv")


