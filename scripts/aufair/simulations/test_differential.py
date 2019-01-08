from aufair import auditing_data as ad
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Ridge, BayesianRidge, LassoLars
from sklearn.svm import SVC
import matplotlib.pyplot as plt


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
    data['x1'] = np.random.normal(size=N) +  data['attr']
    data['x2'] = np.random.normal(size=N) -  data['attr']
    data['noise'] = np.random.normal(scale=0.0, size=N)
    data['y'] =  data['x1'] + data['x1'] + sdata['x2'] + data['noise']  
    data['outcome'] = -1 + 2 * (data.y >= 0).astype('int32')

    # split the data into train versus test set using a 0.7/0.3 ratio
    train = data.loc[np.random.choice(data.index, int(len(data)* 0.7), replace=True), :]
    test = data.drop(train.index)

    # train data is used to fit the audited learner
    features = ['x1', 'x2']
    train_x = np.array(train[features])
    train_y = np.array(train['outcome'].ravel())
    audited = LogisticRegression()
    audited.fit(train_x, train_y)

    # test data is used to audit for fairness
    protected = {'attr': 0}
    test_x = np.array(test[features])
    test['predict'] = audited.predict(test_x)

    # here is where unfairness is created 
    results = pd.DataFrame()
    results.index.name = 'gamma'
    
    for gamma in np.arange(nu_max):
        gamma = gamma / 10 

        # copy test to avoid changes to be compounded
        test_sim = test.copy()
        ind = np.random.choice(test_sim[test_sim['attr'] == 0].index, 
                                int(gamma * len(test_sim[test_sim['attr'] == 0])), 
                                replace=False)
        test_sim.loc[ind, 'predict'] = (-1) * test_sim.loc[ind, 'predict']

        # balance data
        ind1 = np.random.choice(test_sim[test_sim.attr == 0].index, int(0.95*len(test_sim[test_sim.attr == 0])), replace=False)
        ind2 = np.random.choice(test_sim[test_sim.attr == 1].index, int(0.95*len(test_sim[test_sim.attr == 0])), replace=False)
        test_sim = pd.concat([test_sim.loc[ind1, :], test_sim.loc[ind2, :]])
    
        # auditing using a decision tree with x1 and x2
        auditor = DecisionTreeClassifier(max_depth=5)
        features_audit = ['x1', 'x2']
        res = ad.binary_audit(test_sim, features_audit, auditor, protected, seed=10, fraction=1)
        results.loc[gamma,  'unfairness_all'] = res['attr']

    
    return results

def test_experiment2(N, nu_max):
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
    data['x1'] = np.random.normal(size=N)  - 0.0 * (2 *data['attr'] - 1)
    data['x2'] = np.random.normal(size=N) 
    data['noise'] = np.random.normal(scale=0.5, size=N)
    data['y'] =  data['x2'] + data['x1'] + data['noise']  
    data['outcome'] = -1 + 2 * (data.y >= 0).astype('int32')

    # split the data into train versus test set using a 0.7/0.3 ratio
    train = data.loc[np.random.choice(data.index, int(len(data)* 0.7), replace=True), :]
    test = data.drop(train.index)

    # train data is used to fit the audited learner
    features = ['x1', 'x2']
    train_x = np.array(train[features])
    train_y = np.array(train['outcome'].ravel())
    audited = LogisticRegression()
    audited.fit(train_x, train_y)

    # test data is used to audit for fairness
    protected = {'attr': 0}
    test_x = np.array(test[features])
    test['predict'] = audited.predict(test_x)

    # here is where unfairness is created 
    results = pd.DataFrame()
    results.index.name = 'gamma'
    
    for gamma in np.arange(nu_max):
        gamma = gamma / 10 

        # copy test to avoid changes to be compounded
        test_sim = test.copy()
        test_sim.set_index(np.arange(len(test_sim)))
        mask = ((test_sim.attr == 0) & (test_sim.x1 ** 2 + test_sim.x2 ** 2 <= 1) & (test_sim.x1 <= 0) & (test_sim.x2 <= 0)) 
  
        l = len(test_sim[mask])
        ind = np.random.choice(test_sim[mask].index, 
                                int(gamma * l), 
                                replace=False)
        test_sim.loc[ind, 'predict'] = (-1) * test_sim.loc[ind, 'predict']
        
        # auditing using a decision tree with x1 and x2
        auditor = DecisionTreeClassifier(max_depth= 10, min_samples_leaf=0.025)
        audit = ad.detector_data(auditor, test_sim, stepsize=0.025, niter=200)

        feature_list = ['x1']
        balancer = LogisticRegression()
        audit.get_weights(balancer, feature_list, 'attr', 0)

        audit.get_y('predict', 'attr', 0)

        feature_auditing = ['x1', 'x2']
        #g, alpha, test_start = audit.audit_weight(feature_auditing, 'attr', 0, seed=3)
        g, alpha, test_start = audit.audit(feature_auditing, seed=3)
        #g, alpha, test_start = audit.audit_reweight(feature_auditing, 'attr', 0, seed=3)
        test_end = test_start[(test_start.predicted == 1) & (test_start.weight > 0)]
        
        results.loc[gamma, 'expected_gamma'] = 2 * gamma * (test_sim[mask].weight / test_sim.weight.sum()).sum()
        results.loc[gamma,  'unfairness_all'] = g
        results.loc[gamma,  'violations_all'] = alpha

        print((test_end.label * test_end.weight / test_end.weight.sum()).sum())

        
        plt.plot(test_start.x1, test_start.x2, 'r*')
        plt.plot(test_end[test_end.attr != 0].x1, test_end[test_end.attr != 0].x2, 'b*')
        plt.plot(test_end[test_end.attr == 0].x1, test_end[test_end.attr == 0].x2, 'g*')
        
        #res2 = res_alpha.groupby('x2')[['label', 'predicted2', 'predicted']].mean()
        #res2['residuals'] = res2.label - res2.predicted

        #plt.plot(res2.index, res2.label, 'b+')
        #plt.plot(res2.index, res2.predicted2,  'r*')
        #plt.plot(res2.index, res2.residuals,  'g-')
        plt.show()

    return results

if __name__ == "__main__":
    N = 200000
    nu_max = 11
    results = test_experiment2(N, nu_max)
    print(results)
    #results.to_csv("../../results/synth_oracle_exp1.csv")


