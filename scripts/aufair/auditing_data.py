import pandas as pd
import numpy as np
from aufair import auditing as ad
from aufair import mmd
from aufair.mmd_net import MMD
import time

from keras import backend as K

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# remove pandas chained warnings
pd.options.mode.chained_assignment = None 

class detector_data(object):

    def __init__(self, auditor, data, protected, yname, n=None, lw=0.01, stepsize=None, niter=100, min_size=0.01):
        self.data = data
        self.auditor = auditor
        self.stepsize = stepsize
        self.niter = niter
        self.min_size = min_size
        self.protected_attribute = protected[0]
        self.protected_group = protected[1]
        self.yname = yname
        self.lw = lw
        self.n = n

    def get_y(self):
        
        data = self.data.set_index(np.arange(len(self.data)))
        data['weight'] = 1
        data['label'] = data[self.protected_attribute] * data[self.yname]
        self.data = data

    def get_representation(self, model):
        get_representaiton_layer_output = K.function([model.get_layer('input').input],
                                  [model.get_layer('representation').output])
        self.representation = get_representaiton_layer_output
  
    def split_train_test(self, features, seed=None):
        if seed is not None:
            np.random.seed(seed=seed)
        
        if self.n is not None:
            data = self.data.loc[np.random.choice(self.data.index, self.n, replace=False), :]
        else:
            data = self.data
        train = data.loc[np.random.choice(data.index, int(len(data)* 0.7), replace=True), :]
        test = data.drop(train.index)
        return train, test

    def certify(self, features, yname, seed=None, parameter_grid=None):
        train, test = self.split_train_test(features, seed=seed)
        pa = self.protected_attribute
        pg = self.protected_group

        test_x =  np.array(test[features])
        test_y = np.array(test['label']).ravel()
        test_weights = np.array(test.weight).ravel()

        # search for unfairness ceritficate
        detect = ad.detector(self.auditor, niter=self.niter, stepsize=self.stepsize)
        train_x = np.array(train[features])
        train_y = np.array(train['label']).ravel()
        train_weights = np.array(train.weight).ravel()
        

        # compute unfairness level
        detect.certify(train_x, train_y, train_weights)
        test_a = np.array(test[pa]).ravel()
        pred =  np.array(test[yname]).ravel()
        detect.certify(train_x, train_y, train_weights,
                        parameter_grid=parameter_grid)
        # compute unfairness
        predicted = self.auditor.predict(test_x)
        gamma1 = predicted[(test_a == 1) & (predicted == 1) & (pred == 1)].shape[0] / \
                 predicted[(predicted == 1) & (test_a == 1)].shape[0]
        gamma2 = predicted[(test_a == -1) & (predicted == 1) & (pred == 1)].shape[0] / \
                 predicted[(predicted == 1) & (test_a == -1)].shape[0]
        gamma = gamma1 / gamma2
        gamma = (gamma / (1 + gamma) - 0.5) * (predicted[(predicted == 1) & (pred == 1)].shape[0]) / predicted.shape[0]
        #gamma, _ = detect.certificate(test_x, test_y, pred, test_a, test_weights)

        return gamma

    def certify_is(self, features, yname, seed=None):
        train, test = self.split_train_test(features, seed=seed)
        train.set_index(np.arange(len(train)), inplace=True)
        test.set_index(np.arange(len(test)), inplace=True)
        
        pa = self.protected_attribute
        pg = self.protected_group

        train.loc[train[pa] == pg, 'weight'] = (train['w']) / (1-train['w'])
        
        # search for unfairness ceritficate
        detect = ad.detector(self.auditor, niter=self.niter, stepsize=self.stepsize)
        train_x = np.array(train[features])
        train_y = np.array(train['label']).ravel()
        train_weights = np.array(train.weight).ravel()
        detect.certify(train_x, train_y, train_weights)

        # compute unfairness level
        test_x = np.array(test[features])
        test.loc[test[pa] == pg, 'weight'] = (test['w']) / (1-test['w'])
        
        test_x =  np.array(test[features])
        test_y = np.array(test['label']).ravel()
        test_weights = np.array(test.weight).ravel()
        test_a = np.array(test[pa]).ravel()
        pred = np.array(test[yname]).ravel()
        gamma, _ = detect.certificate(test_x, test_y, pred, test_a, test_weights)

        return gamma

    def certify_mmd(self, features, yname, seed=None):
        train, test = self.split_train_test(features, seed=seed)
        train.set_index(np.arange(len(train)), inplace=True)
        test.set_index(np.arange(len(test)), inplace=True)
        
        pa = self.protected_attribute
        pg = self.protected_group

        # mmd method to estimate weights
        X = np.array(train[features])
        A = np.array(train[pa])
        mod = mmd.mmd(lw=self.lw, tol=0.01, learning_rate=0.01)
        mod.fit(X, A)
        train['weight'] = mod.predict(X, A)
        
        test_x = np.array(test[features])
        test_a = np.array(test.attr).ravel()
        test['weight'] = mod.predict(test_x, test_a)

        # search for unfairness ceritficate
        detect = ad.detector(self.auditor, niter=self.niter, stepsize=self.stepsize)
        train_x = np.array(train[features])
        train_y = np.array(train['label']).ravel()
        train_weights = np.array(train.weight).ravel()
        detect.certify(train_x, train_y, train_weights)

        test_x = np.array(test[features])
        test_y = np.array(test['label']).ravel()
        test_weights = np.array(test.weight).ravel()
        test_a = np.array(test[pa]).ravel()
        pred = np.array(test[yname]).ravel()
        gamma, _ = detect.certificate(test_x, test_y, pred, test_a, test_weights)

        return gamma

    def certify_mmd_net(self, features, yname, seed=None, parameter_grid=None):
        train, test = self.split_train_test(features, seed=seed)
        train.set_index(np.arange(len(train)), inplace=True)
        test.set_index(np.arange(len(test)), inplace=True)

        pa = self.protected_attribute
        pg = self.protected_group

        # mmd method to estimate weights
        train['cons'] = 1
        X = np.array(train[features])
        A = np.array(train[pa])
        A = (A + 1) / 2
        mmd = MMD(len(features), target=A, weight=np.array(train.weight).ravel(), lw=self.lw)
        #mod = mmd.model
        mmd.cross_validate_fit(X, A)
        mod = mmd.model
        #mod.fit(X, A, epochs=6, batch_size=512, verbose=0)
        train['wt'] = mod.predict(X)
        train.loc[train[pa] == 1, 'weight'] = train['wt']
        self.get_representation(mod)

        test_a = np.array(test.attr).ravel()
        test_x = np.array(test[features])
        test['wt'] = mod.predict(test_x)
        test.loc[test[pa] == 1, 'weight'] = test['wt']
      
        # search for unfairness ceritficate
        detect = ad.detector(self.auditor, niter=self.niter, stepsize=self.stepsize)
        train_x = self.representation([np.array(train[features]), 1])[0]
        train_y = np.array(train['label']).ravel()
        train_weights = np.array(train.weight).ravel()
        detect.certify(train_x, train_y, train_weights, parameter_grid=parameter_grid)
        train_pred = np.array(train[yname]).ravel()
        train_attr = np.array(train[pa]).ravel()
        gamma, acc = detect.certificate(train_x, train_y, train_pred, train_attr, train_weights)

        test_x = self.representation([np.array(test[features]), 1])[0]
        test_y = np.array(test['label']).ravel()
        test_weights = np.array(test.weight).ravel()
        pred =  np.array(test[yname]).ravel()
        gamma, acc = detect.certificate(test_x, test_y, pred, test_a, test_weights)

        return gamma

    def get_violation(self, features, seed=None):

        train, test = self.split_train_test(features, seed=seed)
        train.set_index(np.arange(len(train)), inplace=True)
        test.set_index(np.arange(len(test)), inplace=True)
        
        pa = self.protected_attribute
        yname = self.yname

        # mmd method to estimate weights
        X = np.array(train[features])
        A = np.array(train[pa])
        A = (A + 1) / 2
        #mmd = MMD(len(features), target=A, weight=np.array(train.weight).ravel(), lw=self.lw)
        #mod = mmd.model
        #mod.fit(X, A, epochs=6, batch_size=512, verbose=0)
        mmd = MMD(len(features), target=A, weight=np.array(train.weight).ravel(), lw=self.lw)
        # mod = mmd.model
        mmd.cross_validate_fit(X, A)
        mod = mmd.model
        train['wt'] = mod.predict(X)
        train.loc[train[pa] == 1, 'weight'] = train['wt']
        self.get_representation(mod)

        # weights for test data
        rep_x = self.representation([np.array(test[features]), 1])[0]
        test_a = np.array(test[pa]).ravel()
        test_x = np.array(test[features])
        test['wt'] = mod.predict(test_x)
        test.loc[test[pa] == 1, 'weight'] = test['wt']

        # get violations
        detect = ad.detector(self.auditor, niter=self.niter, stepsize=self.stepsize)
        train_x = np.array(train[features])
        #train_x = self.representation([np.array(train[features]), 1])[0]
        train_y = np.array(train['label']).ravel()
        train_weights = np.array(train.weight).ravel()
        train_attr = np.array(train[pa]).ravel()
        train_pred = np.array(train[yname]).ravel()
        #detect.violation(train_x, train_y, train_weights, pred, A)
        detect.fit(train_x, train_y, train_weights, train_pred, train_attr)

        # look at test data
        """"
         test_x = self.representation([np.array(test[features]), 1])[0]
        test_y = np.array(test['label']).ravel()
        test_weights = np.array(test.weight).ravel()
        pred = np.array(test[yname]).ravel()
        pred = (pred + 1) / 2
        test_a = (test_a + 1) / 2
        gamma, alpha = detect.delta(test_x, test_y, pred, test_a, test_weights)
        print(gamma)
        print(alpha)
        test['predicted'] = self.auditor.predict(test_x)
       
        """
        test_x = np.array(test[features])
        #test_x = self.representation([np.array(test[features]), 1])[0]
        test_y = np.array(test['label']).ravel()
        test_attr = np.array(test[pa]).ravel()
        test_pred = np.array(test[yname]).ravel()
        gamma, alpha = detect.compute_unfairness(test_x, test_y, test_attr, test_pred)

        self.gamma_array = detect.gamma
        self.alpha_array = detect.alpha

        # extract subgroup
        test['predicted'] = self.auditor.predict(test_x)

        # compute delta
        if gamma > 0:
            delta = np.log(gamma)
        else:
            delta = 0

        return delta, test


    def certify_violation_iter(self, feature, nboot=None, parameter_grid=None):
        results = np.zeros(nboot)
        for iter in np.arange(nboot):
            delta, _ = self.get_violation(feature)
            results[iter] = delta

        return results.mean(axis=0), np.sqrt(results.var())


    def get_violation_individual(self, features, individual, seed=None):

        train, test = self.split_train_test(features, seed=seed)
        train.set_index(np.arange(len(train)), inplace=True)
        test.set_index(np.arange(len(test)), inplace=True)
        
        pa = self.protected_attribute
        pg = self.protected_group
        yname = self.yname

        # mmd method to estimate weights
        X = np.array(train[features])
        A = np.array(train[pa])
        A = (A + 1) / 2
        mmd = MMD(len(features), target=A, weight=np.array(train.weight).ravel(), lw=self.lw)
        mod = mmd.model
        mod.fit(X, A, epochs=6, batch_size=512, verbose=0)
        train['wt'] = mod.predict(X)
        train.loc[train.attr == 1, 'weight'] = train['wt']
        self.get_representation(mod)

        # weights for test data
        test_a = np.array(test.attr).ravel()
        test_x =  np.array(test[features])
        test['wt'] = mod.predict(test_x)
        test.loc[test.attr == 1, 'weight'] = test['wt']

        # get violations
        detect = ad.detector(self.auditor, niter=self.niter, stepsize=self.stepsize)
        train_x = np.array(train[features])
        train_y = np.array(train['label']).ravel()
        train_weights = np.array(train.weight).ravel()
        train_attr = np.array(train[pa]).ravel()
        train_pred = np.array(train[yname]).ravel()
        detect.violation_individual(train_x, train_y, train_weights, train_pred, train_attr, individual)

        # train 2 models
        detect.fit_iter(train_x, train_y, train_weights, detect.eta)

        ## upper bound
        test_x = np.array(test[features])
        test_y = np.array(test['label']).ravel()
        test_attr = np.array(test[pa]).ravel()
        test_pred = np.array(test[yname]).ravel()
        gamma, alpha = detect.compute_unfairness(test_x, test_y, test_attr, test_pred)
        test['predicted'] = self.auditor.predict(test_x)

        ## lower bound
        detect.fit_iter(train_x, train_y, train_weights, detect.eta1)
        gamma1, alpha = detect.compute_unfairness(test_x, test_y, test_attr, test_pred)
        test['predicted1'] = self.auditor.predict(test_x)

        # compute delta
        if gamma > 0:
            delta = np.log(gamma)
        else:
            delta = 0

        if gamma1 > 0:
            delta1 = np.log(gamma1)
        else:
            delta1 = 0

        return delta, delta1, test

    def individual_violation_iter(self, feature, individual, nboot=None):
        results = np.zeros((nboot, 2))
        for iter in np.arange(nboot):
            delta, delta1, _ = self.get_violation_individual(feature, individual)
            results[iter, 0] = delta
            results[iter, 1] = delta1

        return results[:, 0].mean(axis=0), results[:, 1].mean(axis=0)

    def certify_knn(self, features, yname, seed=None):
        train, test = self.split_train_test(features, seed=seed)
        train.set_index(np.arange(len(train)), inplace=True)
        test.set_index(np.arange(len(test)), inplace=True)
        
        pa = self.protected_attribute

        # mmd method to estimate weights
        X = np.array(train[features])
        y = np.array(train[yname]).ravel()
        y = (y + 1)/ 2        
        A = np.array(train[pa])
        A = (A + 1) / 2

        t = 0
        while t < 5:
            # take current weight
            W = np.array(train.weight).ravel()

            # construct network
            mmd = MMD(len(features), target=A, weight=W, lw=self.lw)
            mod = mmd.model_complete
            mod.fit(X, {'fairness': y, 'weight': A}, epochs=5, batch_size=256, verbose=2)

            # predicted weight
            train['wt'] = mod.predict(X)[1].ravel()
            train.loc[train[pa] == 1, 'weight'] = train['wt']
            t += 1

        test_x = np.array(test[features])
        test_a = np.array(test.attr).ravel()
        test_a = (test_a + 1) / 2
        
        predicted = mod.predict(test_x)[0].ravel()
        predicted[predicted < 0.5] = -1
        predicted[predicted >= 0.5] = 1

        # compute unfairness
        pred = np.array(test[yname]).ravel()
        gamma1 = predicted[(test_a == 1) & (predicted == 1) & (pred == 1)].shape[0] / \
                predicted[(predicted == 1) & (test_a == 1)].shape[0]
        gamma2 = predicted[(test_a == 0) & (predicted == 1) & (pred == 1)].shape[0] / \
                 predicted[(predicted == 1) & (test_a == 0)].shape[0]
        gamma = gamma1 / gamma2
        gamma = (gamma / (1 + gamma) - 0.5) * (predicted[(predicted == 1) & (pred == 1)].shape[0]) / predicted.shape[0]

        return gamma

    def certify_iter(self, features, yname, nboot=10, balancing=None, parameter_grid=None):

        results = np.zeros(nboot)
        for iter in np.arange(nboot):
            if balancing is None:
                gamma = self.certify(features, yname, parameter_grid=parameter_grid)
            elif balancing is "IS":
                gamma = self.certify_is(features, yname)
            elif balancing is "MMD":
                gamma = self.certify_mmd(features, yname)
            elif balancing is 'MMD_NET':
                gamma = self.certify_mmd_net(features, yname)
            elif balancing is 'KNN':
                gamma = self.certify_knn(features, yname)
            results[iter] = gamma
            
        return results.mean(axis=0), np.sqrt(results.var())

