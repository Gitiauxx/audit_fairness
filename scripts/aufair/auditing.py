import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from copy import deepcopy

class detector(object):

    def __init__(self, auditor, stepsize=0.01, niter=150, min_size=0.05):
        self.auditor = auditor
        self.stepsize = stepsize
        self.niter = niter
        self.min_size = min_size

    def fit(self, train_x, train_y, train_weights, train_pred, train_attr):


        iter = 0
        self.eta = 0
        eta = 0
        gamma0 = -1

        while (iter < self.niter):
            
            self.fit_iter(train_x, train_y, train_weights, eta)
            gamma, alpha = self.compute_unfairness(train_x, train_y, train_attr, train_pred)

            if gamma < 0:
                gamma = 0

            if np.isnan(gamma):
                self.eta -= self.stepsize
                break
            if alpha < self.min_size:
                self.eta -= self.stepsize
                break
            if np.isnan(alpha):
                self.eta -= self.stepsize
                break
            eta += self.stepsize

            if gamma > gamma0:
                self.eta = eta
                gamma0 = gamma

            iter += 1
 
        # predict subgroup with maximum unfairness
        self.fit_iter(train_x, train_y, train_weights, self.eta)

    def violation_individual(self, train_x, train_y, train_weights, train_pred, train_attr, x):

        iter = 0
        self.eta = 0
        eta = 0
        tag = 0
        outcome = 1
        alpha = 0
        alpha_max = 0

        while (iter < self.niter) & (alpha_max - alpha < 0.025):

            self.fit_iter(train_x, train_y, train_weights, eta)
            gamma, alpha = self.compute_unfairness(train_x, train_y, train_attr, train_pred)

            if gamma < 0:
                gamma = 0

            if (outcome == -1) & (tag == 0):
                self.eta1 = self.eta - self.stepsize
                alpha_max = alpha
                tag = 1

            if np.isnan(gamma):
                self.eta -= self.stepsize
                break
            if alpha < self.min_size:
                self.eta -= self.stepsize
                break
            if np.isnan(alpha):
                self.eta -= self.stepsize
                break
            eta += self.stepsize
            self.eta = eta

            outcome = self.auditor.predict(x.reshape(1, -1))


            iter += 1

        if tag == 0:
            self.eta1 = self.eta


        """

        eta = self.eta + 0.5
        self.eta1 = eta
        iter = 0
        gamma0 = np.inf
        outcome = -1

        while (iter < self.niter) & (outcome == -1):

            self.fit_iter(train_x, train_y, train_weights, eta)
            gamma, alpha = self.compute_unfairness(train_x, train_y, train_attr, train_pred)
            eta -= self.stepsize

            if gamma <= gamma0:
                self.eta1 = eta
                gamma0 = gamma

            outcome = self.auditor.predict(x.reshape(1, -1))

            iter += 1
    """

    def fit_iter(self, train_x, train_y, weights, eta):
        
        # change weight
        weights_adjusted = weights.copy().astype(float)
        weights_adjusted[(train_y == -1)] = 1.0 * weights_adjusted[train_y == -1] * (1.0 + eta)

        # train auditor
        self.auditor.fit(train_x, train_y, 
                        sample_weight=weights_adjusted)

    def compute_unfairness(self, X, y, attr, pred):
        
        # multi differential unfairness
        predicted = self.auditor.predict(X)
        indicator = (predicted + 1) / 2
       

        if predicted[(predicted == 1) & (attr == 1)].shape[0] > 0:
            gamma = predicted[(attr == 1) & (predicted == 1) & (pred == 1)].shape[0]/ predicted[(predicted == 1) & (attr == 1)].shape[0]

        else:
            gamma = np.nan

        if predicted[(predicted == 1) & (attr == -1) &(pred == 1)].shape[0] > 0:
            gamma2 = predicted[(attr == -1) & (predicted == 1) & (pred == 1)].shape[0] / predicted[(predicted == 1) & (attr == -1)].shape[0]
            gamma = gamma / gamma2

        else:
            gamma = np.nan
       
        # compute size of violation
        alpha = predicted[predicted == 1].shape[0] / predicted.shape[0]

        return gamma, alpha

    def certify(self, X, y, weights, parameter_grid=None):

        if parameter_grid is None:
            self.auditor.fit(X, y, sample_weight=weights)
        else:

            # hyperparameter tuning using a k-fold validation technique and random grid search
            k = parameter_grid['cv']
            grid_searched = parameter_grid['parameter']
            niter = parameter_grid['niter']

            auditor_random = RandomizedSearchCV(estimator = self.auditor, 
                                param_distributions = grid_searched, 
                                n_iter = niter, 
                                cv = k,
                                scoring='accuracy')
            auditor_random.fit(X, y, sample_weight=weights)
            self.auditor = auditor_random.best_estimator_
            self.auditor.fit(X, y, sample_weight=weights)

    def certificate(self, X, y, pred, A, weights):
        predicted = self.auditor.predict(X)
        accuracy = weights[predicted == y].sum() / weights.sum()
    
        attr = weights[A == pred].sum() / weights.sum()
        alpha = weights[predicted == y].sum() / weights.sum()

        return (accuracy - 1 + attr) / 4, alpha

    def delta(self, X, y, pred, A, weights, threshold=None):


        if threshold is None:
            predicted = self.auditor.predict(X)
        else:
            predict_proba = self.auditor.predict_proba(X)[:, 1]
            predicted = (predict_proba > threshold).astype('int32')

        alpha = predicted[(predicted == 1) & (y == 1)].shape[0] / predicted.shape[0]

        d = predicted[(predicted == 1) & (pred == 1) & (A == 1)].shape[0]
        if predicted[(predicted == 1) & (pred == 1)].shape[0] == 0:
            return 0, alpha
        
        d = d / predicted[(predicted == 1) & (pred == 1)].shape[0]
        
        #d2 = predicted[(predicted == 1) & (pred == 1) & (A == 0)].shape[0]
        #print(d2)
        #if predicted[(predicted == 1) & (A == 0)].shape[0] == 0:
            #return 0, 0
        
        #d2 = d2 / predicted[(predicted == 1) & (pred == 1)].shape[0]
       
        if d < 1:
            return np.log(d / (1 - d)), alpha
        else:
            return 0, 0


    def violation(self, X, y, weights, pred, A):

        iter = 0
        self.eta = 0
        eta = 0
        gamma0 = -1
        gamma = 0
        w = deepcopy(weights)
        
        while (iter < self.niter) & (gamma > gamma0 - 0.1) :
            
            # certificate
            self.certify(X, y, w)
            gamma, alpha = self.delta(X, y, pred, A, weights)

            w[ y == -1] = weights[y == -1] * (1 + eta)
            if np.isnan(gamma) | np.isnan(alpha) | (alpha < self.min_size):
                break
            if gamma > gamma0:
                self.eta = eta
                gamma0 = gamma

            eta += self.stepsize
            iter += 1

        weights[y == - 1] = weights[y == -1] * (1 + self.eta)
        self.certify(X, y, weights)
        gamma, alpha = self.delta(X, y, pred, A, weights)

    def violation_boosting(self, X, y, weights, pred, A):

        alpha = 1
        iter = 1
        mod_dict = {}
        w0 = deepcopy(weights / weights.sum())

        # Initialize model
        self.certify(X, y, w0)
        weak_auditor = deepcopy(self.auditor)
        predicted = weak_auditor.predict(X)
        # compute group size
        alpha = weights[predicted == 1].shape[0] / weights.shape[0]
        print(alpha)

        error = w0[predicted == y].sum() / w0.sum()
        weight_model = 1
        mod_dict[0] = (weak_auditor, weight_model)

        w = w0 * np.exp(- weight_model * y * predicted)


        while alpha > self.min_size:

            # compute current prediction
            predicted_model = np.zeros(y.shape[0])
            for i in mod_dict.keys():
                mod = mod_dict[i]

                predicted_model += mod[1] * (mod[0].predict(X))

            predicted_model = 2 * (predicted_model > 0).astype('int32') - 1

            # compute group size
            alpha = weights[predicted_model == 1].shape[0] / weights.shape[0]

            # computer new weights
            w = w / w.sum()
            w1 = w * (self.stepsize * (1 - y) / 2 + (1 - self.stepsize) * (1 + y) / 2)
            #w = w / w.sum()

            # compute weak auditor
            self.certify(X, y, w1)
            weak_auditor = deepcopy(self.auditor)

            # compute model weight
            predicted = weak_auditor.predict(X)
            error = w[predicted != y].sum()/ w.sum()
            print(alpha)


            if error < 0.5:
                weight_model = 1/2 * np.log((1 - error) / error)

            else:
                break

            # update weights
            w = w * np.exp(-weight_model * predicted * y)

            # add weak auditor
            mod_dict[iter] = (weak_auditor, weight_model)

            iter += 1

