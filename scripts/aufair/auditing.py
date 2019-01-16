import numpy as np

class detector(object):

    def __init__(self, auditor, stepsize=0.01, niter=100, min_size=0.02):
        self.auditor = auditor
        self.stepsize = stepsize
        self.niter = niter
        self.min_size = min_size

    def fit(self, train, features, yname, protected_attribute):

        # create input/output
        train_x = np.array(train[features])
        train_y = np.array(train['label']).ravel()
        train_weights = np.array(train.weight).ravel()
        train_attr = np.array(train[protected_attribute]).ravel()
        train_pred = np.array(train[yname]).ravel()
        
        iter = 0
        self.eta = 0
        eta = 0
        gamma0 = -1
        gamma = 0
        
        while (iter < self.niter) & (gamma >= gamma0 - 0.1):
            
            self.fit_iter(train_x, train_y, train_weights, eta)
            gamma, alpha = self.compute_unfairness(train_x, train_y, train_attr, train_pred)
           
            predicted = self.auditor.predict(train_x)
            if predicted[(predicted == -1 ) & (train_attr == -1)].shape[0]:
                a = predicted[(predicted == 1 ) & (train_attr == -1)].shape[0] / predicted[(predicted == -1 ) & (train_attr == -1)].shape[0]
            else:
                a = 10000
            #gamma = np.log( gamma / (1 - gamma) * a)
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
        

    def fit_iter(self, train_x, train_y, weights, eta):
        
        # change weight
        weights_adjusted  = weights.copy().astype(float)
        weights_adjusted[(train_y == -1) ] = 1.0 * weights_adjusted[train_y == -1] * (1.0 + eta)

        #weights_adjusted[(pred == -1) & (attr == 1) ] = weights[(pred == -1) & (attr == 1) ] * (1 + eta)
        #weights_adjusted[(pred == 1) & (attr == -1) ] = weights[(pred == 1) & (attr == -1) ] * (1 + eta)
       
        # train auditor
        self.auditor.fit(train_x, train_y, 
                        sample_weight=weights_adjusted)

    def compute_unfairness(self, X, y, attr, pred):
        
        # multi differential unfairness
        predicted = self.auditor.predict(X)
        indicator = (predicted + 1) / 2
       

        if predicted[(predicted == 1) & (pred == 1)].shape[0] > 0:
            gamma = predicted[(attr == 1) & (predicted == 1) & (pred == 1)].shape[0]/ predicted[(predicted == 1) & (pred == 1)].shape[0]
        else:
            gamma = np.nan
       
        # compute size of violation
        alpha = predicted[predicted == 1].shape[0] / predicted.shape[0]

        return gamma, alpha





