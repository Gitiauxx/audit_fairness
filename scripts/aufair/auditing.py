import numpy as np

class detector(object):

    def __init__(self, auditor, stepsize=0.01, niter=100, min_size=0.05):
        self.auditor = auditor
        self.stepsize = stepsize
        self.niter = niter
        self.min_size = min_size

    def fit(self, train_x, train_y, weights):
        
        iter = 0
        self.eta = 0
        eta = self.eta 
        gamma0 = -1
        
        while (iter < self.niter):
            self.fit_iter(train_x, train_y, weights, eta)
            gamma, alpha = self.compute_unfairness(train_x, train_y, weights)
            
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
        print(alpha)
        # predict subgroup with maximum unfairness
        self.fit_iter(train_x, train_y, weights, self.eta)
        

    def fit_iter(self, train_x, train_y, weights, eta):
        
        # change weight
        weights_adjusted  = weights.copy()
        weights_adjusted[train_y == -1] = weights[train_y == -1] * (1 + eta)
       
        # train auditor
        self.auditor.fit(train_x, train_y, 
                        sample_weight=weights_adjusted)

    def compute_unfairness(self, X, y, weights):
        
        # multi differential unfairness
        predicted = self.auditor.predict(X)
        indicator = (predicted == 1).astype('int32')
        weights_filtered = weights[predicted == 1]
        indicator_filtered = indicator[predicted == 1]
        y_filtered = y[predicted == 1]

        gamma = np.abs(np.inner(indicator_filtered * weights_filtered / \
                weights_filtered.sum(), y_filtered))
        
        # compute size of violation
        alpha = (weights_filtered / weights.sum()).sum()

        return gamma, alpha





