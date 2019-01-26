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

def test_experiment(n, n_test, nu_max, auditor, sigma_noise=0.0, unbalance=0, nboot=10):
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
    data['attr'] = np.random.choice([-1, 1], n)
    data['x1'] = np.random.normal(size=n)  - unbalance *data['attr']
    data['x2'] = np.random.normal(size=n) 
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
    results.index.name = 'gamma'

    for gamma in np.arange(0, nu_max):
        gamma = gamma / 10 
      
        # copy test to avoid changes to be compounded
        test_sim = test.copy()
        test_sim.set_index(np.arange(len(test_sim)))
        
        # define violation shape
        mask = test_sim.x1**2 + test_sim.x2**2 <= 1
        mask = (test_sim.x1 <= -0) & (test_sim.x2 <= -0)
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
        audit = ad.detector_data(auditor, test_sim, stepsize=0.01, niter=500)
        audit.get_y('predict', 'attr', 1)

        feature_auditing = ['x1', 'x2']
        g, g_std = audit.audit_iter(features, 'predict', 'attr', nboot=nboot)
        
        results.loc[gamma, 'delta'] = l  / (l + l2 * (1-gamma))
        results.loc[gamma, 'delta'] = np.log(results.loc[gamma, 'delta'] / (1 - results.loc[gamma, 'delta']))
        results.loc[gamma,  'estimated_delta'] = g
        results.loc[gamma,  'delta_deviation'] = g_std

        #test_end = test_start[(test_start.predicted == 1)]
        #plt.plot(test_sim.x1, test_sim.x2, 'r*')
        #plt.plot(test_end.x1, test_end.x2, 'b*')
        #plt.show()
    
    return results

def test_unbalance(n, n_test, nu_max, auditor, sigma_noise=0.2, unbalance=0, nboot=10):
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
    data['attr'] = np.random.choice([-1, 1], n)
    data['x1'] = np.random.normal(size=n)  + unbalance * data['attr']
    data['x2'] = np.random.normal(size=n) - unbalance * data['attr']
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
    results.index.name = 'gamma'

    for gamma in np.arange(0, nu_max):
        gamma = gamma / 10 
      
        # copy test to avoid changes to be compounded
        test_sim = test.copy()
        test_sim.set_index(np.arange(len(test_sim)))
        
        # define violation shape
        mask = test_sim.x1**2 + test_sim.x2**2 <= 1
        #mask = (test_sim.x1 <= -0) & (test_sim.x2 <= -0)
        mask1 = ((test_sim.attr == 1)  & (mask))
        mask2 = ((test_sim.attr == -1) & (mask))
        
        np.random.seed(seed=1)
        l = len(test_sim[mask1])
        ind = np.random.choice(test_sim[mask1].index, 
                                int(l), 
                                replace=False)

        test_sim.loc[ind, 'predict'] = (-1) * test_sim.loc[ind, 'predict']

        np.random.seed(seed=1)
        l2 = len(test_sim[mask2])
        ind = np.random.choice(test_sim[mask2].index, 
                                int((1-gamma) * l2), 
                                replace=False)

        test_sim.loc[ind, 'predict'] = (-1) * test_sim.loc[ind, 'predict']

        # auditing using a decision tree with x1 and x2
        audit = ad.detector_data(auditor, test_sim, stepsize=0.025, niter=500)
        audit.get_y('predict', 'attr', 1)

        feature_auditing = ['x1', 'x2']
        #delta, _, test_start = audit.audit_weight2(features, 'predict', 'attr', 1, nu= 10)
        #delta, _ = audit.audit_iter_weight(features, 'predict', 'attr', 1, nu= 10, nboot=nboot)
        #delta2, _ = audit.audit_iter_weight(features, 'predict', 'attr', 1, nu= 30, nboot=nboot)
        delta, alpha = audit.cross_validation(features, 'predict', 'attr', 1)
        delta2, _, _ = audit.audit_weight2(features, 'predict', 'attr', 1, nu= 10)
    
        l1 = len(test_sim[(mask1) & (test_sim.x1 <= 0) & (test_sim.x2 <= 0)])
        l2 = len(test_sim[(mask2) & (test_sim.x1 <= 0) & (test_sim.x2 <= 0)])
        l3 = len(test_sim[(mask1) & (test_sim.predict == 1) & (test_sim.x1 <= 0) & (test_sim.x2 <= 0)])
        l4 = len(test_sim[(mask2) & (test_sim.predict == 1) & (test_sim.x1 <= 0) & (test_sim.x2 <= 0)])
        ll = l3  / (l4)

        ll2 = l1 / l2
        results.loc[gamma, 'delta'] = np.log(ll / ll2)
        results.loc[gamma,  'estimated_delta_10'] = delta
        results.loc[gamma,  'estimated_delta_30'] = delta2
        """
        test_end = test_start[(test_start.predicted == 1)]
        plt.plot(test_sim.x1, test_sim.x2, 'r*')
        plt.plot(test_end[test_end.attr == 1].x1, test_end[test_end.attr == 1].x2, 'b*')
        plt.plot(test_end[test_end.attr == -1].x1, test_end[test_end.attr == -1].x2, 'g*')
        plt.show()
        """
        
    
    
    return results

def test_certifying(n, n_test, nu_max, auditor, sigma_noise=0.2, unbalance=0, nboot=10):
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
    data['attr'] = np.random.choice([-1, 1], n)
    data['x1'] = np.random.normal(size=n)  - unbalance *data['attr']
    data['x2'] = np.random.normal(size=n) 
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
    results.index.name = 'gamma'

    for gamma in np.arange(0, nu_max):
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
        audit = ad.detector_data(auditor, test_sim, stepsize=0.01, niter=0)
        audit.get_y('predict', 'attr', 1)

        feature_auditing = ['x1', 'x2']
        g = audit.certify(features, 'predict',  seed=1)
        
        results.loc[gamma, 'delta'] = l  / len(test_sim[test_sim.predict == 1]) - 1/2 * (l + (1-gamma) * l2) / len(test_sim[test_sim.predict == 1])
        #results.loc[gamma, 'delta'] = np.log(results.loc[gamma, 'delta'] / (1 - results.loc[gamma, 'delta']))
        results.loc[gamma,  'estimated_delta'] = g
        #results.loc[gamma,  'delta_deviation'] = g_std
    
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


