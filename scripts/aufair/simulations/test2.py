from aufair import auditing_data as ad
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Ridge, BayesianRidge, LassoLars
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

def test_certifying(n, n_test, nu_min, nu_max, auditor, sigma_noise=0.2, unbalance=0, nboot=10, balancing=None, lw=10**(-4)):
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
    data['noise'] = np.random.normal(scale=0.2, size=n)
    data['w'] =  np.exp(unbalance * data['x2'] - unbalance * data['x1'] + data['noise'] )
    data['w'] = data['w'] / (1 + data['w'])
    data['u'] = np.random.uniform(0, 1, size=len(data))
    data.loc[data.u < data.w, 'attr'] = -1
    data.loc[data.u >= data.w, 'attr'] = 1
    
    data['noise'] = np.random.normal(scale=sigma_noise, size=n)
    data['y'] =  data['x2'] + data['x1'] + data['noise']  
    data['outcome'] = - 1 + 2 * (data.y >= 0).astype('int32')

    # split the data into train versus test set using a 0.7/0.3 ratio
    train = data.loc[np.random.choice(data.index, int(len(data)* 0.7), replace=True), :]
    test = data.drop(train.index)
    test = test.loc[np.random.choice(test.index, n_test, replace=False), :]

    # train data is used to fit the audited learner
    features = ['x1', 'x2']
    train_x = np.array(train[features])
    train_y = np.array(train['outcome'].ravel())
    audited = LogisticRegression()
    audited.fit(train_x, train_y)

    # test data is used to audit for fairness
    protected = {'attr': 1}
    test_x = np.array(test[features])
    test['predict'] = audited.predict(test_x)

    # here is where unfairness is created 
    results = pd.DataFrame()
    results.index.name = 'nu'

    for gamma in np.arange(nu_min, nu_max):
        gamma = gamma / 10 
      
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

        l2 = len(test_sim[mask2])
        ind = np.random.choice(test_sim[mask2].index, 
                                int((1-gamma) * l2), 
                                replace=False)

        test_sim.loc[ind, 'predict'] = (-1) * test_sim.loc[ind, 'predict']

        # auditing using a decision tree with x1 and x2
        protected = ('attr', 1)
        yname = 'predict'
        audit = ad.detector_data(auditor, test_sim, protected, yname, lw=lw, niter=0)
        audit.get_y()

        feature_auditing = ['x1', 'x2']
        g, g_std = audit.certify_iter(features, 'predict',  nboot=nboot, balancing=balancing)

        if gamma < 1:
            delta = l / (l + (1 - gamma) * l2) 
            delta = delta / (1 - delta)
            delta = delta / l * l2
            delta = (delta - 1) / (2 * (delta + 1)) * (l + (1-gamma) * l2) / len(test_sim)
        else:
            delta = 1/2 * (l + (1-gamma) * l2) / len(test_sim)
        
        results.loc[gamma, 'gamma'] = delta
        results.loc[gamma,  'estimated_gamma'] = g
        results.loc[gamma,  'gamma_deviation'] = g_std
        results.loc[gamma, 'bias'] = g  / results.loc[gamma, 'gamma']
    
    return results


if __name__ == "__main__":
    n = 500000
    n_test = 15000
    nu_max = 10
    auditor = SVC()
    auditor = DecisionTreeClassifier(max_depth=5, min_samples_leaf=0.02)
    #auditor = RandomForestClassifier(n_estimators=20, max_depth=2)
    #auditor = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    #results = test_unbalance(n, n_test, nu_max, auditor, nboot=10, unbalance=0.2)
    results = test_certifying(n, n_test, nu_max, auditor, nboot=10, unbalance=0.0)
    print(results)
    #results.to_csv("../../results/synth_oracle_exp1.csv")


