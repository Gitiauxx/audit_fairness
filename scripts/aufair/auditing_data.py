import pandas as pd
import numpy as np
from aufair import auditing as ad
import time

from sklearn.linear_model import LogisticRegression

# remove pandas chained warnings
pd.options.mode.chained_assignment = None 

class detector_data(object):

    def __init__(self, auditor, data, stepsize=0.01, niter=100, min_size=0.01):
        self.data = data
        self.auditor = auditor
        self.stepsize = stepsize
        self.niter = niter
        self.min_size = min_size

    def get_y(self, yname, protected_attribute, protected_group):
        data = self.data
        data['weight'] = 1
        data['label'] = data[protected_attribute] 
        #data = data[data[yname] == 1]
        self.data = data
  
    def split_train_test(self, features, seed=None):
        if seed is not None:
            np.random.seed(seed=seed)
        
        data = self.data
        train = data.loc[np.random.choice(data.index, int(len(data)* 0.7), replace=True), :]
        test = data.drop(train.index)

        return train, test

    def audit(self, features, yname, seed=None):
        train, test = self.split_train_test(features, seed=seed)
        test = test[test[yname] == 1]
        train = train[train[yname] == 1]

        detect = ad.detector(self.auditor, niter=self.niter, stepsize=self.stepsize)

        train_x = np.array(train[features])
        train_y = np.array(train['label']).ravel()
        train_weights = np.array(train.weight).ravel()
        detect.fit(train_x, train_y, train_weights)
    
        # compute unfairness
        test_x =  np.array(test[features])
        test_y = np.array(test['label']).ravel()
        test_weights = np.array(test.weight).ravel()
        gamma, alpha = detect.compute_unfairness(test_x, test_y, test_weights)

        # extract subgroup
        test['predicted'] = self.auditor.predict(test_x)

        # compute delta
        if gamma < 1:
            delta = np.log(gamma / (1 - gamma))
        else:
            delta = 0

        return delta, alpha, test

    def audit_iter(self, features, yname, nboot=100):

        results = np.zeros((nboot, 2))
        for iter in np.arange(nboot):
            seed = int(time.time())
            delta, alpha, _ = self.audit(features, yname, seed=seed)

            results[iter, 0] = delta
            results[iter, 1] = alpha

        return results[:, 0].mean(), np.sqrt(results[:, 0].var())

    def audit_weight(self, features, yname, protected_attribute, protected_group, nu= 10, max_iter=10, seed=None):

        train, test = self.split_train_test(features, seed=seed)
        train.set_index(np.arange(len(train)), inplace=True)
        train_x =  np.array(train[features])

        adj0 = 100
        delta = 0
        delta0 = 0
        beta = 0
        adj = 0
        

        t = 0
        detect = ad.detector(self.auditor, niter=self.niter, stepsize=self.stepsize)

        while (t < max_iter)   :

            train_bal = train.copy(deep=True)
            
            train_bal = train_bal[train_bal[yname] == 1]

            X =  np.array(train_bal[features])
            y = np.array(train_bal['label']).ravel()

            weights =  np.array(train_bal.weight)    
            detect.fit(X, y, weights)

            score = self.auditor.predict_proba(train_x)[:, 1]
            train['score'] = score
        
            score_quantile = train.score.quantile([i / nu for i in np.arange(nu)])
        
            train['bin'] = 0
            b = 1
            for q in score_quantile:
                train.loc[train.score >= q, 'bin'] = b
                b += 1

            weight_updated = train[train[protected_attribute] != protected_group].groupby('bin')['bin'].size().to_frame('w1')
            weight_updated['w2'] = train[train[protected_attribute] == protected_group].groupby('bin')['bin'].size()
            train = pd.merge(train, weight_updated, left_on='bin', right_index=True, how='left')
            
            train.loc[train[protected_attribute] == protected_group, 'weight'] = train['weight'] * 0.25 + 0.75 * train['w1'] / train['w2']
            print(train[(train.x1 > 1.5) & (train.x2 < 0)].weight.describe())
            train.drop(['w1', 'w2'], axis=1, inplace=True)

            # look at progress
            train['predicted'] = self.auditor.predict(train_x)
            adj =  train[(train.predicted == 1) & (train[protected_attribute] != protected_group)].weight.sum() / \
                            train[(train.predicted == 1) & (train[protected_attribute] == protected_group)].weight.sum()
            print(adj)

            t += 1

        
        # compute unfairness on test data
        test_x =  np.array(test[features])
        test_y = np.array(test['label']).ravel()
        test['predicted'] = self.auditor.predict(test_x)
        adj =  len(test[(test.predicted == 1) & (test[protected_attribute] != protected_group)]) / \
                            len(test[(test.predicted == 1) & (test[protected_attribute] == protected_group)])
        
        test = test[test[yname] == 1]
        test_x =  np.array(test[features])
        test_y = np.array(test['label']).ravel()
        test_weights = np.array(test.weight).ravel()
        gamma, alpha = detect.compute_unfairness(test_x, test_y, test_weights)

        # extract subgroup
        test['predicted'] = self.auditor.predict(test_x)
        
        # compute delta
        if gamma < 1:
            delta = np.log(adj * gamma / (1 - gamma))
        else:
            delta = 0

        return delta, alpha, test


    def audit_weight2(self, features, yname, protected_attribute, protected_group, nu= 0.1, max_iter=10, seed=None):

        train, test = self.split_train_test(features, seed=seed)
        train.set_index(np.arange(len(train)), inplace=True)
        
        train_x = np.array(train[features])
        train_y = np.array(train['label']).ravel()
        weights = np.array(train.weight).ravel()
        detect = ad.detector(self.auditor, niter=self.niter, stepsize=self.stepsize)
        
        #detect.fit(train_x, train_y, weights)
        balancer = LogisticRegression()
        balancer.fit(train_x, train_y)
        #train['predicted'] = self.auditor.predict(train_x)
        #train['weight0'] = balancer.predict_proba(train_x)[:, 1]
        #train.loc[train[protected_attribute]  == protected_group, 'weight'] = (1 - train['weight0']) / train['weight0']
        #train['weight'] = train.weight0 * (1 - train.weight0)
        print(train.weight.describe())
        #train.loc[train.weight > 5, 'weight'] = 5
        
        #er = train[train.predicted == train[protected_attribute]].weight.sum() / train.weight.sum()
        #er = -1/2 * np.log((1-er) / er)
        
        #train.loc[train[protected_attribute] == protected_group, 'weight'] = train['weight']  * \
                  #            np.exp(nu * (1 - train['predicted']) / 2)
        #train['weight'] = train.weight / train.weight.sum()

        train_bal = train.copy(deep=True)
        train_bal = train_bal[train_bal[yname] == 1]

        t = 0

    
        while (t < 1)   :
                        
            # compute unfairness for train_bal
            X = np.array(train_bal[features])
            y = np.array(train_bal['label']).ravel()
            weights = np.array(train_bal.weight).ravel()
        
            detect.fit(X, y, weights)

            # look how the auditor predict protected group
            train_bal['predicted'] = self.auditor.predict(X)
            train_bal.loc[train_bal[protected_attribute] != protected_group, 'weight'] = train_bal['weight']  * \
                                np.exp(-nu * (train_bal[protected_attribute] == train_bal['predicted']).astype('int32'))
            
            # see progress
            train['predicted'] = self.auditor.predict(train_x)
            adj = len(train[(train.predicted == 1) & (train[protected_attribute] == protected_group)]) / \
                            len(train[(train.predicted == 1) & (train[protected_attribute] != protected_group)])
            
            #adj = train[(train.predicted == 1) & (train[protected_attribute] == protected_group)].weight.sum() / \
                #train[(train.predicted == 1) & (train[protected_attribute] != protected_group)].weight.sum()
            
            delta, _= detect.compute_unfairness(X, y, weights)
            delta = adj * delta / (1 - delta)

            print(adj)
            
            

            t += 1

        # compute unfairness on test data
        test_x =  np.array(test[features])
        test_y = np.array(test['label']).ravel()
        test['predicted'] = self.auditor.predict(test_x)
        adj =  len(test[(test.predicted == 1) & (test[protected_attribute] != protected_group)]) / \
                            len(test[(test.predicted == 1) & (test[protected_attribute] == protected_group)])
        
        test = test[test[yname] == 1]
        test_x =  np.array(test[features])
        test_y = np.array(test['label']).ravel()
        test_weights = np.array(test.weight).ravel()
        gamma, alpha = detect.compute_unfairness(test_x, test_y, test_weights)

        # extract subgroup
        test['predicted'] = self.auditor.predict(test_x)
        
        # compute delta
        if gamma < 1:
            delta = gamma / (1 - gamma)
            # adjust for unbalance
            delta = delta 
            delta = delta * adj
        else:
            delta = 0

        return delta, alpha, test
