import numpy as np

class mmd(object):

    def __init__(self, lw=0.1, learning_rate=0.001, tol=0.001, conv=10**(-8)):

        self.lw = lw
        self.learning_rate =learning_rate
        self.conv = conv
        self.tol = tol
    
    def discrepancy(self, X, A, W):
        
        n1 = A[A==1].shape[0]
        n2 = A[A==-1].shape[0]
        
        L = np.dot(X[A==1].transpose(), W[A==1]).transpose() / n1
        R = X[A==-1].sum(axis=0) / n2
        return L - R

    def norm_discrepancy(self, F):
        return np.linalg.norm(F)

    def gradient(self, X, A, W, beta):

        F = self.discrepancy(X, A, W)
       
        XA = X[A == 1] 
        XW = np.multiply(XA, W[A == 1][:, np.newaxis])
        DW = np.dot(XA.transpose(), XW)
        DW = np.dot(DW, F)
        DW = DW / XA.shape[0]
        
        return 2 * DW

    def gradient_sigmoid(self, X, A, W, beta):

        F = self.discrepancy(X, A, W)
        FF = self.norm_discrepancy(F)

        XA = X[A == 1] 
        Z = np.multiply(W, 1 - W)
        XW = np.multiply(XA, Z[A == 1][:, np.newaxis])
        DW = np.dot(XA.transpose(), XW)
        DW = np.dot(DW, F)
        DW = DW / XA.shape[0]
        DW = DW / FF
        
        return  DW

    def fit(self, X, A):
        self.beta = np.zeros(X.shape[1])
        loss = np.inf
        loss0 = 0
        self.W = np.ones(X.shape[0])
        t = 0
        
        while (loss > self.tol) & (np.abs(loss - loss0) > self.conv) & (t < 200):

            self.W[A == 1] = np.exp(np.multiply(X[A == 1], self.beta).sum(axis=1))
            
            #self.W[A == 1] = self.W[A == 1] / (1 + 1/2 * self.W[A == 1])
            self.beta = self.beta - \
                        (self.learning_rate * self.gradient(X, A, self.W, self.beta) + self.lw * self.beta)
            
            loss0 = loss
            loss = self.discrepancy(X, A, self.W)
            loss = np.linalg.norm(loss)
            self.loss = loss
            t += 1
            
    def predict(self, X, A):
        W = np.ones(X.shape[0])
        W[A == 1] = np.exp(np.multiply(X[A == 1], self.beta).sum(axis=1))
        return W

if __name__ == "__main__":
    import pandas as pd

    n = 5000
    unbalance = 0.2

    data = pd.DataFrame(index=np.arange(n))
    data['attr'] = np.random.choice([-1, 1], n)
    data['x1'] = np.random.normal(size=n) 
    data['x2'] = np.random.normal(size=n) 
    data['noise'] = np.random.normal(scale=0.2, size=n)
    data['y'] =  np.exp(0.5 * data['x2'] - 0.5 * data['x1'] + data['noise'] )
    data['y'] = data['y'] / (1 + data['y'])
    data['u'] = np.random.uniform(0, 1, size=len(data))
    data.loc[data.u < data.y, 'attr'] = -1
    data.loc[data.u >= data.y, 'attr'] = 1

    # split train and test
    train = data.loc[np.random.choice(data.index, int(len(data)* 0.7), replace=True), :]
    test = data.drop(train.index)


    mmd_estimator = mmd(lw=0.15, tol=0.1, learning_rate=0.005)
    X = np.array(train[['x1', 'x2']])
    A = np.array(train.attr).ravel()
    mmd_estimator.fit(X, A)

    test_x = np.array(test[['x1', 'x2']])
    test_a = np.array(test.attr).ravel()
    test['weight'] = mmd_estimator.predict(test_x, test_a)

    mask = (test.x1**2 + test.x2**2 <= 1)
    test = test[mask]
    print(test.loc[test.attr == 1, 'weight'].sum() / len(test[test.attr ==1]) - 
    test.loc[test.attr == -1, 'weight'].sum() / len(test[test.attr==-1]))

        
