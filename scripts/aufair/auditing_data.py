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
        gamma, _ = detect.certificate(test_x, test_y, pred, test_a, test_weights)

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
        test_x =  np.array(test[features])
        test.loc[test[pa] == pg, 'weight'] = (test['w']) / (1-test['w'])
        
        test_x =  np.array(test[features])
        test_y = np.array(test['label']).ravel()
        test_weights = np.array(test.weight).ravel()
        test_a = np.array(test[pa]).ravel()
        pred =  np.array(test[yname]).ravel()
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

        test_x =  np.array(test[features])
        test_y = np.array(test['label']).ravel()
        test_weights = np.array(test.weight).ravel()
        test_a = np.array(test[pa]).ravel()
        pred =  np.array(test[yname]).ravel()
        gamma, _ = detect.certificate(test_x, test_y, pred, test_a, test_weights)

        return gamma

    def certify_mmd_net(self, features, yname, seed=None):
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
        mod = mmd.model
        mod.fit(X, A, epochs=6, batch_size=512, verbose=0)
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
        detect.certify(train_x, train_y, train_weights)
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
        mmd = MMD(len(features), target=A, weight=np.array(train.weight).ravel(), lw=self.lw)
        mod = mmd.model
        mod.fit(X, A, epochs=6, batch_size=512, verbose=0)
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
        train_x = self.representation([np.array(train[features]), 1])[0]
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
        test_x = self.representation([np.array(test[features]), 1])[0]
        test_y = np.array(test['label']).ravel()
        test_attr = np.array(test[pa]).ravel()
        test_pred = np.array(test[yname]).ravel()
        gamma, alpha = detect.compute_unfairness(test_x, test_y, test_attr, test_pred)
     

        # extract subgroup
        test['predicted'] = self.auditor.predict(test_x)

        # compute delta
        if gamma > 0:
            delta = np.log(gamma)
        else:
            delta = 0

        return delta, test

        return gamma, test

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
        pg = self.protected_group

        # mmd method to estimate weights
        X = np.array(train[features])
        y = np.array(train[yname]).ravel()
        y = (y + 1)/ 2        
        A = np.array(train[pa])
        A = (A + 1) / 2
        

        t = 0
        while t < 1:
            W = np.array(train.weight).ravel()
            mmd = MMD(len(features), target=A, weight=W, lw=self.lw)
            mod = mmd.model_complete
            mod.fit({'input': X[A==1][:5000, :], 'input2': X[A == 0][:5000, :]}, {'fairness': y[A== 1][:5000]}, 
                    epochs=10, batch_size=32, verbose=2)
            self.get_representation(mod)
            rep_x = self.representation([X[A==1][:5000, :], 1])[0]
            rep_x1 = self.representation([X[A==-1][:5000, :], 1])[0]
            print(K.eval(mmd.kera_cost(rep_x, rep_x1)(A, W)))
            #train['wt'] = mod.predict(X)[0].ravel()
            #train.loc[train.attr == 1, 'weight'] = (1 - train['wt']) /  train['wt']
            t += 1
        
        
        test_x = np.array(test[features])
        test_a = np.array(test.attr).ravel()
        test_y = np.array(test[yname]).ravel()
        
        predicted = mod.predict({'input': test_x[test_a == 1][:1500, :], 'input2': test_x[test_a == -1][:1500, :]})
        #test['wt'] = predicted[0]
        #test.loc[test.attr == 1, 'weight'] = (1 - test['wt']) /  test['wt']

        #self.get_representation(mod)
        #rep_x = self.representation([test_x, 1])[0]
        #print(K.eval(mmd.kera_cost(rep_x)((test_a + 1) / 2, np.array(test.weight))))

        # compute unfairness
        weights = np.array(test.weight).ravel()
        pred =  np.array(test[yname]).ravel()
        score = 2 * (predicted.ravel() >= 0.5).astype('int32') - 1
        print(score)
        print(score[score == test_y[test_a == 1][:1500]].shape[0] / score.shape[0])
        accuracy = weights[score == test_y].sum() / weights.sum()
        attr = weights[test_a == pred].sum() / weights.sum()
        gamma = (2 * accuracy - 1 ) / 4
    
        return gamma
        
    def certify_quant(self, features, yname, seed=None, model=None, nu=5):

        train, test = self.split_train_test(features, seed=seed)
        train.set_index(np.arange(len(train)), inplace=True)
        test.set_index(np.arange(len(test)), inplace=True)

        pa = self.protected_attribute
        pg = self.protected_group

        # estimate importance sampling weights
        if model is None:
            mod = LogisticRegression()
        
        X = np.array(train[features])
        y = np.array(train[pa])
        mod.fit(X, y)
        train['score'] = mod.predict_proba(X)[:, 1]

        # create quantile estimate of weights
        score_quantile = train.score.quantile([i / nu for i in np.arange(nu)])
        train['bins'] = 0

        b = 0
        for q in score_quantile:
            train.loc[train.score >= q, 'bins'] = b
            b += 1

        weight_updated = train[train[pa] != pg].groupby('bins')['bins'].size().to_frame('w1')
        weight_updated['w2'] = train[train[pa] == pg].groupby('bins')['bins'].size()
        train = pd.merge(train, weight_updated, left_on='bins', right_index=True, how='left')
            
        train.loc[train[pa] == pg, 'weight'] = train.loc[train[pa] == pg, 'w1'] / train.loc[train[pa] == pg, 'w2']
        train.drop(['w1', 'w2'], axis=1, inplace=True)

        # search for unfairness ceritficate
        detect = ad.detector(self.auditor, niter=self.niter, stepsize=self.stepsize)
        train_x = np.array(train[features])
        train_y = np.array(train['label']).ravel()
        train_weights = np.array(train.weight).ravel()
        detect.certify(train_x, train_y, train_weights)

        # compute unfairness level
        test_x =  np.array(test[features])
        test['score'] = mod.predict_proba(test_x)[:, 1]
        test['bins'] = 0

        b = 0
        for q in score_quantile:
            test.loc[test.score >= q, 'bins'] = b
            b += 1

        weight_updated = test[test[pa] != pg].groupby('bins')['bins'].size().to_frame('w1')
        weight_updated['w2'] = test[test[pa] == pg].groupby('bins')['bins'].size()
        test = pd.merge(test, weight_updated, left_on='bins', right_index=True, how='left')

        test.loc[test[pa] == pg, 'weight'] = test.loc[test[pa] == pg, 'w1'] / test.loc[test[pa] == pg, 'w2']
        test.drop(['w1', 'w2'], axis=1, inplace=True)
       
        test_x =  np.array(test[features])
        test_y = np.array(test['label']).ravel()
        test_weights = np.array(test.weight).ravel()
        test_a = np.array(test[pa]).ravel()
        pred =  np.array(test[yname]).ravel()
        gamma, acc, attr = detect.certificate(test_x, test_y, pred, test_a, test_weights)

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

