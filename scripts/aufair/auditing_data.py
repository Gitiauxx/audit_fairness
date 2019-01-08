import pandas as pd
import numpy as np
from aufair import auditing as ad
import time

# remove pandas chained warnings
pd.options.mode.chained_assignment = None 

class detector_data(object):

    def __init__(self, auditor, data, stepsize=0.01, niter=100, min_size=0.01):
        self.data = data
        self.auditor = auditor
        self.stepsize = stepsize
        self.niter = niter
        self.min_size = min_size

    def get_weights(self, balancer, features, protected_attribute, protected_group):
        data = self.data
        data['weight'] = 1
        
        X = np.array(data[features])
        y = np.array(data[protected_attribute]).ravel()

        balancer.fit(X, y)
        weights = balancer.predict_proba(X)
        
        data['weight0'] = weights[:, 1]
        data.loc[data.attr != protected_group, 'weight'] = (1 - data['weight0']) / \
                                          data['weight0']
        
        #clipping the weight 
        data.loc[data.weight0 < 0.1, 'weight'] = 0
        data.loc[data.weight0 > 0.9, 'weight'] = 0
        data = data[data.weight > 0]
        data['weight'] = 1
        data.loc[data.attr != protected_group, 'weight'] = 1.5 * data.loc[data.attr != protected_group, 'weight']

        self.data = data
        
        #data['weight'] = data.weight * data.weight0 * (1 - data.weight0)
        #data['weight'] = data.weight / data.weight.sum()
        #data = data[(data.weight0 > 0.01) & (data.weight0 < 0.99)]
        
        #data.drop('weight0', axis=1, inplace=True)

    def get_y(self, yname, protected_attribute, protected_group):
        data = self.data
        data['label'] = data[yname] * (data[protected_attribute] == protected_group).astype('int32') - \
                data[yname] * (data[protected_attribute] != protected_group).astype('int32')
  
    def split_train_test(self, features, seed=None):
        if seed is not None:
            np.random.seed(seed=seed)
        
        data = self.data
        train = data.loc[np.random.choice(data.index, int(len(data)* 0.7), replace=True), :]
        test = data.drop(train.index)

        return train, test

    def audit(self, features, seed=None):
        train, test = self.split_train_test(features, seed=seed)

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

        return gamma, alpha, test

    def audit_weight(self, features, protected_attribute, protected_group, conv=20, seed=None):

        train, test = self.split_train_test(features, seed=seed)
        detect = ad.detector(self.auditor, niter=self.niter, stepsize=self.stepsize)

        train_x = np.array(train[features])
        train_y = np.array(train['label']).ravel()
        train = train.set_index(np.arange(len(train)))

        test_x = np.array(test[features])
        test_y = np.array(test['label']).ravel()
        
        iteration = 0
        diff_w0 = 0
        diff_w = 1
        
        while (iteration < conv) & (np.abs(diff_w - diff_w0) > 0.001):
            train_weights = np.array(train.weight).ravel()
            detect.fit(train_x, train_y, train_weights)
            train['predicted'] = self.auditor.predict(train_x)
    
            # compute difference in distribtion
            diff_w0 = diff_w
            
            diff_w = train[(train.predicted == 1) & \
                          (train[protected_attribute] == protected_group)].weight.sum()
            diff_w = diff_w / train[(train.predicted == 1) & \
                            (train[protected_attribute] != protected_group)].weight.sum()
            print(diff_w)
            
            train.loc[(train.predicted == 1) & \
                          (train[protected_attribute] == protected_group), 'weight'] = \
                          train.loc[(train.predicted == 1) & \
                          (train[protected_attribute] == protected_group), 'weight'] / diff_w
            """
            test['predicted'] = self.auditor.predict(test_x)
            test.loc[(test.predicted == 1) & \
                          (test[protected_attribute] == protected_group), 'weight'] = \
                          test.loc[(test.predicted == 1) & \
                          (test[protected_attribute] == protected_group), 'weight'] / diff_w
            """
            diff_w2 = train[(train.predicted != 1) & \
                          (train[protected_attribute] == protected_group)].weight.sum()
            diff_w2 = diff_w2 / train[(train.predicted != 1) & \
                            (train[protected_attribute] != protected_group)].weight.sum()
            print(diff_w2)
            
            train.loc[(train.predicted != 1) & \
                          (train[protected_attribute] == protected_group), 'weight'] = \
                          train.loc[(train.predicted != 1) & \
                          (train[protected_attribute] == protected_group), 'weight'] / diff_w2

            """
            test.loc[(test.predicted != 1) & \
                          (test[protected_attribute] == protected_group), 'weight'] = \
                          test.loc[(test.predicted != 1) & \
                          (test[protected_attribute] == protected_group), 'weight'] / diff_w2
            """

            iteration += 1

        # compute unfairness
        test_weights = np.array(test.weight).ravel()
        gamma, alpha = detect.compute_unfairness(test_x, test_y, test_weights)

        # extract subgroup
        test['predicted'] = self.auditor.predict(test_x)
        gamma = gamma * test[(test.predicted == 1) & (test[protected_attribute] != protected_group)].weight.sum()
        gamma = gamma / test[(test.predicted == 1) & (test[protected_attribute] == protected_group)].weight.sum()
        

        return gamma, alpha, test

    def audit_reweight(self, features, protected_attribute, protected_group, conv=20, seed=None):

        train, test = self.split_train_test(features, seed=seed)
        detect = ad.detector(self.auditor, niter=self.niter, stepsize=self.stepsize)

        train_x = np.array(train[features])
        train_y = np.array(train['label']).ravel()
        train = train.set_index(np.arange(len(train)))

        test_x = np.array(test[features])
        test_y = np.array(test['label']).ravel()
        
        iteration = 0
        gamma0 = 1
        diff_w = 1

        train_weights0 = np.array(train.weight).ravel()
        
        while (iteration < conv):
            train_weights = np.array(train.weight).ravel()
            detect.fit(train_x, train_y, train_weights)
            train['predicted'] = self.auditor.predict(train_x)
            gamma, _ = detect.compute_unfairness(train_x, train_y, train_weights0)
    
            # compute size of both subgroups
            size1 = len(train[(train.predicted == 1) & \
                          (train[protected_attribute] == protected_group)])
            size2 = len(train[(train.predicted == 1) & \
                          (train[protected_attribute] != protected_group)])
         
            if size2 < 0.1 * (size1 + size2):
                train.loc[(train.predicted == 1) & \
                          (train[protected_attribute] == protected_group), 'weight'] = \
                          train.loc[(train.predicted == 1) & \
                          (train[protected_attribute] == protected_group), 'weight'] / 1.5
            
            elif size1 < 0.1 * (size1 + size2):
                train.loc[(train.predicted == 1) & \
                          (train[protected_attribute] != protected_group), 'weight'] = \
                          train.loc[(train.predicted == 1) & \
                          (train[protected_attribute] != protected_group), 'weight'] / 1.5

            else:
                break
            diff_w = np.abs(gamma - gamma0)
            gamma0 = gamma
            iteration += 1

        # compute unfairness
        test_weights = np.array(test.weight).ravel()
        gamma, alpha = detect.compute_unfairness(test_x, test_y, test_weights)
        
        # extract subgroup
        test['predicted'] = self.auditor.predict(test_x)
        gamma = gamma * test[(test.predicted == 1) & (test[protected_attribute] == protected_group)].weight.sum()
        gamma = gamma / test[(test.predicted == 1) & (test[protected_attribute] != protected_group)].weight.sum()
        

        return gamma, alpha, test
            

    def audit_iter(self, features, nboot=100):

        results = np.zeros((nboot, 2))
        for iter in np.arange(nboot):
            seed = int(time.time())
            gamma, alpha, _ = self.audit(features, seed=seed)
            results[iter, 0] = gamma
            results[iter, 1] = alpha

        return results 
