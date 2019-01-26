import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

class dataset(object):

    def __init__(self, sigma_noise=0.2, unbalance=0, n=5000):

        self.sigma_noise = sigma_noise
        self.unbalance = unbalance
        self.n = n
        
        
    def make_data(self, n):

        # simulate a synthethic data
        data = pd.DataFrame(index=np.arange(n))
        data['attr'] = np.random.choice([-1, 1], n)
        data['x1'] = np.random.normal(size=n)  - self.unbalance *data['attr']
        data['x2'] = np.random.normal(size=n) 
        data['noise'] = np.random.normal(scale=self.sigma_noise, size=n)
        data['y'] =  data['x2'] + data['x1'] + data['noise']  
        data['outcome'] = - 1 + 2 * (data.y >= 0).astype('int32')

        self.data = data
        self.protected_attribute = 'attr'
        self.protected_group = 1

    
    def classify(self, audited=None):

        if audited is None:
            audited = LogisticRegression()
        # train data is used to fit the audited learner
        features = ['x1', 'x2']
        X = np.array(self.data[features])
        y = np.array(self.data['outcome'].ravel())
        audited.fit(X, y)

        # test data is used to audit for fairness
        test_x = np.array(self.data[features])
        self.data['predict'] = audited.predict(test_x)
        self.yname = 'predict'

    def get_y(self, data):

        data['weight'] = 1
        data['label'] = data[self.protected_attribute] * data[self.yname]
        return data

    def simulate_unfairness(self, nu, alpha):

        nu = nu / 20
        gamma = 4 * nu / (2 * nu + 1)
        alpha = 2 * alpha / (1 + 1 - gamma)
      
        # copy test to avoid changes to be compounded
        data = self.data.set_index(np.arange(len(self.data)))
        
        # define violation shape
        mask = (data.x1**2 + data.x2**2 <= 1) & (data.predict == -1)
        mask1 = ((data.attr == 1)  & (mask))
        mask2 = ((data.attr == -1) & (mask))
        
        np.random.seed(seed=1)
        l = len(data[mask1])
        ind = np.random.choice(data[mask1].index, 
                                int(l), 
                                replace=False)

        data.loc[ind, 'predict'] = (-1) * data.loc[ind, 'predict']

        l2 = len(data[mask2])
        ind = np.random.choice(data[mask2].index, 
                                int((1-gamma) * l2), 
                                replace=False)

        data.loc[ind, 'predict'] = (-1) * data.loc[ind, 'predict']
    
        # assemble data
        N = (1 - alpha) / alpha * len(data.loc[mask])
        data1 = data.loc[mask, :]
        
        data = data.drop(data1.index)
        ind = np.random.choice(data.index, int(N), replace=True)
        data2 = data.loc[ind, :]
        data = pd.concat([data1, data2])
        data.set_index(np.arange(len(data)), inplace=True)

        return data

    def split_train_test(self, data, seed=None):
        if seed is not None:
            np.random.seed(seed=seed)
        
        data = data.loc[np.random.choice(data.index, self.n, replace=False), :]
        train = data.loc[np.random.choice(data.index, int(len(data)* 0.7), replace=True), :]
        test = data.drop(train.index)
        return train, test




