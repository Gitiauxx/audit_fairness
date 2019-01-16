import pandas as pd
import numpy as np
from aufair import auditing as ad
import time

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

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
        self.data = data
  
    def split_train_test(self, features, seed=None):
        if seed is not None:
            np.random.seed(seed=seed)
        
        data = self.data
        train = data.loc[np.random.choice(data.index, int(len(data)* 0.7), replace=True), :]
        test = data.drop(train.index)

        return train, test

    def audit(self, features, yname, protected_attribute, seed=None):
        train, test = self.split_train_test(features, seed=seed)
        train = train[train[yname] == 1]
        test = test[test[yname] == 1]
       
        detect = ad.detector(self.auditor, niter=self.niter, stepsize=self.stepsize)

        train_x = np.array(train[features])
        train_y = np.array(train['label']).ravel()
        train_weights = np.array(train.weight).ravel()
        detect.fit(train, features, yname, protected_attribute)
    
        # compute unfairness
        test_x =  np.array(test[features])
        test_y = np.array(test['label']).ravel()
        test_attr = np.array(test[protected_attribute]).ravel()
        test_pred = np.array(test[yname]).ravel()
        gamma, alpha = detect.compute_unfairness(test_x, test_y, test_attr, test_pred)

        # extract subgroup
        test['predicted'] = self.auditor.predict(test_x)

        # compute delta
        if gamma < 1:
            delta = np.log(gamma / (1 - gamma))
        else:
            delta = 0

        return delta, alpha, test

    def audit_iter(self, features, yname, protected_attribute, nboot=100):

        results = np.zeros((nboot, 2))
        for iter in np.arange(nboot):
            seed = int(time.time())
            delta, alpha, _ = self.audit(features, yname, protected_attribute, seed=seed)

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
            b = 0
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


    def audit_weight2(self, features, yname, protected_attribute, protected_group, nu= 10, seed=None):

        train, test = self.split_train_test(features, seed=seed)
        train.set_index(np.arange(len(train)), inplace=True)
        
        train_x = np.array(train[features])
        train_y = np.array(train['label']).ravel()
        train_attr = np.array(train[protected_attribute]).ravel()
        weights = np.array(train.weight).ravel()
        detect = ad.detector(self.auditor, niter=self.niter, stepsize=self.stepsize)
        
        balancer = LogisticRegression(solver="lbfgs")
        balancer.fit(train_x, train_attr)
        train['score'] = balancer.predict_proba(train_x)[:, 1]
        
        
        score_quantile = train.score.quantile([i / nu for i in np.arange(nu)])
        train['bins'] = 0

        b = 0
        for q in score_quantile:
            train.loc[train.score >= q, 'bins'] = b
            b += 1

        #train = train[train.bins < train.bins.max()]
        #train = train[train.bins > train.bins.min()]

        weight_updated = train[train[protected_attribute] != protected_group].groupby('bins')['bins'].size().to_frame('w1')
        weight_updated['w2'] = train[train[protected_attribute] == protected_group].groupby('bins')['bins'].size()
        train = pd.merge(train, weight_updated, left_on='bins', right_index=True, how='left')
            
        train.loc[train[protected_attribute] == protected_group, 'weight'] = train['w1'] / train['w2']
        #( 1 - train.score) / train.score
        #rain['w1'] / train['w2']
        train.drop(['w1', 'w2'], axis=1, inplace=True)

        #train = train[train[yname] == 1]

        chosen_bin = 10
        delta = -0.1
        delta0 = -1
        

        features1 = features + ['score']
        train = train[train[yname] == 1]
        train_x = np.array(train[features1])

        detect = ad.detector(self.auditor, niter=self.niter, stepsize=self.stepsize)
        detect.fit(train, features1, yname, protected_attribute)

        train['predicted'] = self.auditor.predict(train_x)
        a  = train[(train.predicted == 1) & (train[protected_attribute] == protected_group)].weight.mean()
       
       
        # compute unfairness on test data
        test_x =  np.array(test[features])
        test['score'] = balancer.predict_proba(test_x)[:, 1]
        test = test[test[yname] == 1]
        test_x =  np.array(test[features1])
        
        test_y = np.array(test['label']).ravel()
    
        test_attr = np.array(test[protected_attribute]).ravel()
        test['predicted'] = self.auditor.predict(test_x)
        if test.predicted.mean() == -1:
            return 0, 0, test

    
        test_pred = np.array(test[yname]).ravel()
        gamma, alpha = detect.compute_unfairness(test_x, test_y, test_attr, test_pred)
        
        if len(test[(test.predicted == 1) & (test[yname]== 1)]) > 0:
            gamma = len(test[(test.predicted == 1) & (test.attr == 1) & (test[yname] == 1)]) / len(test[(test.predicted == 1) & (test[yname] == 1)])
        
        elif len(test[(test.predicted == 1) & (test[yname] == -1)]) > 0:
            gamma = len(test[(test.predicted == 1) & (test.attr == 1) & (test[yname] == -1)]) / len(test[(test.predicted == 1) & (test[yname] == -1)])
        
        else:
            gamma = 1
      
        # compute delta
        if gamma < 1:
            delta = gamma / (1 - gamma)
            # adjust for unbalance
            delta = delta 
            delta = np.log(delta * a)
        else:
            delta = 0

        return delta, alpha, test

    def audit_iter_weight(self, features, yname, protected_attribute, protected_group, nboot=100,  nu=10):

        results = np.zeros((nboot, 2))
        for iter in np.arange(nboot):
            #seed = int(time.time())
            delta, alpha, _ = self.audit_weight2(features, yname, protected_attribute, protected_group, nu=nu)

            results[iter, 0] = delta
            results[iter, 1] = alpha

        return results[:, 0].mean(), np.sqrt(results[:, 0].var())


    def cross_validation(self, features, yname, protected_attribute, protected_group):

        nu_current = 0
        current_improvement = -1


        for nu in np.arange(5, 15):
            delta, alpha, df = self.audit_weight2(features, yname, protected_attribute, protected_group, nu=nu)

            train = df.loc[np.random.choice(df.index, int(len(df)* 0.7), replace=True), :]
            test = df.drop(train.index)

            train_x = np.array(train[features])
            train_y = np.array(train[protected_attribute]).ravel()

            test_x = np.array(test[features])
            test_y = np.array(test[protected_attribute]).ravel()

            mod = LogisticRegression(solver='lbfgs')

            mod.fit(train_x, train_y)
            predicted = mod.predict(test_x)
            accuracy0 = predicted[predicted == test_y].shape[0] / predicted.shape[0]

            train_x = np.array(train[features + ['predicted']])
            test_x = np.array(test[features + ['predicted']])
            mod.fit(train_x, train_y)
            predicted = mod.predict(test_x)
            accuracy1 = predicted[predicted == test_y].shape[0] / predicted.shape[0]

            improvement = accuracy1 - accuracy0
            print(improvement)
           
            if improvement > current_improvement:
                current_improvement = improvement
                nu_current = nu

        
        delta, alpha, df = self.audit_weight2(features, yname, protected_attribute, protected_group, nu=nu_current)
        return delta, alpha

