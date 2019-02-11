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

def test_certifying(n, n_test, nu_min, nu_max, auditor, alpha=0.1, sigma_noise=0.0, unbalance=0, nboot=10, parameter_grid=None, balancing=None, stepsize=0.01):
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
    data['noise'] = np.random.normal(scale=sigma_noise, size=n)
    
    # create weights
    data['w'] = np.exp(unbalance * (data['x2'] + data['x1']) ** 2 )
    data['w'] = data['w'] / (1 + data['w'])
    data['w'] = 0.5
   
    data['u'] = np.random.uniform(0, 1, size=len(data))
    data.loc[data.u < data.w, 'attr'] = 1
    data.loc[data.u >= data.w, 'attr'] = -1

    # outcome
    data['noise'] = np.random.normal(scale=sigma_noise, size=n)
    data['y'] = (data['x2'] + data['x1'] + data['noise']) ** 3
    #data['x2'] + data['x1'] + data['noise']
    data['outcome'] = - 1 + 2 * (data.y >= 0).astype('int32')

    # split the data into train versus test set using a 0.7/0.3 ratio
    train = data.loc[np.random.choice(data.index, int(len(data)* 0.7), replace=True), :]
    test = data.drop(train.index)
    test = test.loc[np.random.choice(test.index, n_test, replace=False), :]
    test = data

    # train data is used to fit the audited learner
    features = ['x1', 'x2']
    train_x = np.array(train[features])
    train_y = np.array(train['outcome'].ravel())
    audited = LogisticRegression(solver='lbfgs')
    audited.fit(train_x, train_y)

    # test data is used to audit for fairness
    protected = {'attr': 1}
    test_x = np.array(test[features])
    test['predict'] = audited.predict(test_x)
    test = test.set_index(np.arange(len(test)))

    # here is where unfairness is created 
    results = pd.DataFrame()
    results.index.name = 'nu'

    for nu in np.arange(nu_min, nu_max):
        print(nu)
        nu = nu / 20
        gamma = 4 * nu / (2 * nu + 1)
        alpha1 = 2 * alpha / (1 + 1 - gamma)
      
        # copy test to avoid changes to be compounded
        test_sim = test.copy()
        test_sim.set_index(np.arange(len(test_sim)))
        
        # define violation shape
        mask = (test_sim.x1**2 + test_sim.x2**2 <= 1) & (test_sim.predict <= 0)
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

        # second violation
        # define violation shape
        lmask = (test_sim.x1 ** 2 + test_sim.x2 ** 2 > 1) & (test_sim.x1 ** 2 + test_sim.x2 ** 2 <= 2) & (test_sim.predict <= 0)
        # mask = (test_sim.x1 <= -0) & (test_sim.x2 <= -0)
        lmask1 = ((test_sim.attr == 1) & (lmask))
        lmask2 = ((test_sim.attr == -1) & (lmask))

        np.random.seed(seed=1)
        ll = len(test_sim[lmask1])
        ind = np.random.choice(test_sim[lmask1].index,
                               int(ll),
                               replace=False)

        test_sim.loc[ind, 'predict'] = (-1) * test_sim.loc[ind, 'predict']

        ll2 = len(test_sim[lmask2])
        ind = np.random.choice(test_sim[lmask2].index,
                               int((1 - gamma / 2) * ll2),
                               replace=False)

        test_sim.loc[ind, 'predict'] = (-1) * test_sim.loc[ind, 'predict']

        # construct data
        N = (1 - alpha1) / alpha1 * len(test_sim.loc[(mask) | (lmask)])
        test1 = test_sim.loc[mask, :]
        test_sim = test_sim.drop(test1.index)
        ind = np.random.choice(test_sim.loc[lmask, :].index, len(test1), replace=True)
        test2 = test_sim.loc[ind, :]
        d = len(test2[(test2.predict ==1) & (test2.attr == 1)])/ len(test2[(test2.predict ==1) & (test2.attr == -1)])
        dd = len(test2[(test2.attr == 1)])/ len(test2[(test2.attr == -1)])
        print(np.log(d / dd))
        test_sim = test_sim.drop(test2.index)
        ind = np.random.choice(test_sim.index, int(N), replace=True)
        test3 = test_sim.loc[ind, :]
        test_sim = pd.concat([test1, test2, test3])

        # auditing using a decision tree with x1 and x2
        protected = ('attr', 1)
        yname = 'predict'
        audit = ad.detector_data(auditor, test_sim, protected, yname, n=n_test, stepsize=stepsize, niter=300)
        audit.get_y()
        individual = np.array([-0.95, -0.95])

        feature_auditing = ['x1', 'x2']
        """
        delta, g_std = audit.certify_violation_iter(features, nboot=nboot)
            #audit.audit_iter(features, yname, 'attr', nboot=nboot)
    
        results.loc[gamma, 'delta'] = np.log(l / ((1 - gamma) * l2))
        results.loc[gamma,  'estimated_delta'] = delta
        results.loc[gamma,  'gamma_deviation'] = g_std
        results.loc[gamma, 'bias'] = delta - results.loc[gamma, 'delta']
        """
        delta, delta1 = audit.individual_violation_iter(features, individual, nboot=nboot)
            #audit.get_violation_individual(features, individual, seed=1)


        results.loc[gamma, 'delta'] = np.log(ll / ((1 - gamma / 2) * ll2))
        results.loc[gamma, 'delta_wv'] = np.log(l / ((1 - gamma) * l2))
        results.loc[gamma, 'upper_delta'] = delta
        results.loc[gamma, 'lower_delta'] = delta1


        """
        plt.plot(test_final.x1, test_final.x2, 'b*')
        plt.plot(test_final[test_final.predicted1 == 1].x1, test_final[test_final.predicted1 == 1].x2, 'g*')
        plt.plot(test_final[test_final.predicted == 1].x1, test_final[test_final.predicted == 1].x2, 'r*')
        plt.plot(individual[0], individual[1], "o")
        plt.show()
        """

    
    return results


if __name__ == "__main__":
    n = 500000
    n_test = 5000
    nu_max = 10
    auditor = SVC()
    auditor = DecisionTreeClassifier(max_depth=5, min_samples_leaf=0.02)
    #auditor = RandomForestClassifier(n_estimators=20, max_depth=2)
    #auditor = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    #results = test_unbalance(n, n_test, nu_max, auditor, nboot=10, unbalance=0.2)
    results = test_certifying(n, n_test, nu_max, auditor, nboot=10, unbalance=0.0)
    print(results)
    #results.to_csv("../../results/synth_oracle_exp1.csv")


