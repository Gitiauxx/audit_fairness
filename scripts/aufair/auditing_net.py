import pandas as pd
import numpy as np
from aufair import auditing as ad
from aufair import mmd
from aufair.mmd_net import MMD

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import backend as K
from keras import optimizers

class audit_net(object):

    def __init__(self, data, protected, yname, n_features, niter=1):
        self.data = data
        self.protected_attribute = protected[0]
        self.protected_group = protected[1]
        self.yname = yname
        self.n_features = n_features
        self.auditor = self.create_model()
        self.niter = niter

    def get_y(self):
        data = self.data
        data['weight'] = 1
        data['label'] =  (data[self.protected_attribute] * data[self.yname] + 1) / 2
        self.data = data

    def get_weight(self, model):
        get_weight_layer_output = K.function([model.get_layer('input').input],
                                  [model.get_layer('weight').output])
        self.weight_layer = get_weight_layer_output

    def get_representation(self):
        model = self.auditor
        get_representaiton_layer_output = K.function([model.get_layer('input').input],
                                  [model.get_layer('representation').output])
        self.representation = get_representaiton_layer_output


    def create_model(self):
        
        # create model
        model = Sequential()
        model.add(Dense(8, input_dim=self.n_features, activation='elu', name='input'))
        model.add(Dense(8, activation='elu'))
        model.add(Dropout(0.2))
        model.add(Dense(2, activation='elu', name='representation'))
        model.add(Dropout(0.2))
        model.add(Dense(8, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))

        # register model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.save_weights('model.h5')
        return model 

    def split_train_test(self, features, seed=None):
        if seed is not None:
            np.random.seed(seed=seed)
        
        data = self.data.loc[np.random.choice(self.data.index, 15000, replace=False), :]
        train = data.loc[np.random.choice(data.index, int(len(data)* 0.7), replace=True), :]
        test = data.drop(train.index)

        return train, test

    def certify(self, features, yname, seed=None):
        train, test = self.split_train_test(features, seed=seed)
        pa = self.protected_attribute
        pg = self.protected_group

        # search for unfairness ceritficate
        X = np.array(train[features])
        y = np.array(train['label']).ravel()
        w = np.array(train.weight).ravel()

        self.auditor.load_weights('model.h5')
        self.auditor.fit(X, y, sample_weight=w, batch_size=128, verbose=0, epochs=2)
        
        # measure unfairness on test data
        test_x =  np.array(test[features])
        test_y = np.array(test['label']).ravel()
        test_weights = np.array(test.weight).ravel()
        test_a = np.array(test[pa]).ravel()
        pred =  np.array(test[yname]).ravel()
        gamma = self.certificate(test_x, test_y, pred, test_a, test_weights)

        print(gamma)

        return gamma

    def certify_rep(self, features, yname, seed=None):

        train, test = self.split_train_test(features, seed=seed)
        train.set_index(np.arange(len(train)), inplace=True)
        test.set_index(np.arange(len(test)), inplace=True)

        pa = self.protected_attribute
        pg = self.protected_group

        # search for unfairness ceritficate
        X = np.array(train[features])
        y = np.array(train['label']).ravel()
        A = np.array(train[pa])
        w = np.array(train.weight).ravel()
        pred =  np.array(train[yname]).ravel()

        # estimate weights
        mod = mmd.mmd(lw=0.001, tol=0.001, learning_rate=0.05)
        mod.fit(X, A)
        print(mod.beta)
        train['weight'] = mod.predict(X, A)
        t = 0

        while ( t < self.niter):

            # inputs
            X = np.array(train[features])
            y = np.array(train['label']).ravel()
            A = np.array(train[pa])
            w = np.array(train.weight).ravel()
            pred =  np.array(train[yname]).ravel()

            # search for unfairness ceritficate
            self.auditor.load_weights('model.h5')
            self.auditor.fit(X, y, sample_weight=w, batch_size=128, epochs=5, verbose=0)
            gamma = self.certificate(X, y, pred, A, w)

            # extract representation
            self.get_representation()
            PX = np.array(self.representation([X, 1]))[0]
            train['p1']  = PX[:, 0]
            train['p2']  = PX[:, 1]
    
            # estimate weights
            #mod.fit(X, A)
            #train['weight'] = mod.predict(X, A)
            #train.loc[train[pa] == pg, 'weight'] = train.loc[train[pa] == pg, 'ww']
            
            t += 1

        
        test_x = np.array(test[features])
        test_a = np.array(test.attr).ravel()

        test_PX = np.array(self.representation([test_x, 1]))[0]
        test['ww'] = mod.predict(test_x, test_a)
        test.loc[test[pa] == pg, 'weight'] = test.loc[test[pa] == pg, 'ww']
        test_weights = np.array(test.weight)
        test_y = np.array(test['label']).ravel()
        pred =  np.array(test[yname]).ravel()
        gamma = self.certificate(test_x, test_y, pred, test_a, test_weights)
        print(gamma)
        print('end')


        return gamma

    def certify_knn(self, features, yname, seed=None):

        train, test = self.split_train_test(features, seed=seed)
        train.set_index(np.arange(len(train)), inplace=True)
        test.set_index(np.arange(len(test)), inplace=True)

        pa = self.protected_attribute
        pg = self.protected_group

        # search for unfairness ceritficate
        X = np.array(train[features])
        y = np.array(train['label']).ravel()
        A = np.array(train[pa]).ravel()
        w = np.array(train.weight).ravel()
        pred =  np.array(train[yname]).ravel()
    
        # construct model
        source = X
        target = (A == 1).astype('int32')
        mmd = MMD(source, target, lw=0.001, tol=0.001)
        model = mmd.create_complete_model()
        self.auditor = model
        model.fit(X, [y, target], epochs=4, batch_size=24, verbose=2)

        # compute unfairness
        test_x = np.array(test[features])
        test_a = np.array(test.attr).ravel()
        test_y = np.array(test['label']).ravel()
        
        self.get_weight(model)
        test['ww']  = np.exp(np.array(self.weight_layer([test_x, 1]))[0])
        test.loc[test[pa] == pg, 'weight'] = test.loc[test[pa] == pg, 'ww']


        predicted = model.predict(test_x)[1].ravel()
        predicted = (predicted >  0.5).astype('int32')
        print(predicted[predicted == test_y].shape[0]/ predicted.shape[0])

        test['predicted'] = predicted
        print(test.head(10))
        test_weights = np.array(test.weight).ravel()     
        
        pred =  np.array(test[yname]).ravel()
        gamma = self.certificate(test_x, test_y, pred, test_a, test_weights)
        print(gamma)

        return gamma



    def certify_dnn(self, features, yname, seed=None):

        train, test = self.split_train_test(features, seed=seed)
        train.set_index(np.arange(len(train)), inplace=True)
        test.set_index(np.arange(len(test)), inplace=True)

        pa = self.protected_attribute
        pg = self.protected_group

        # search for unfairness ceritficate
        X = np.array(train[features])
        y = np.array(train['label']).ravel()
        A = np.array(train[pa])
        w = np.array(train.weight).ravel()
        pred =  np.array(train[yname]).ravel()

        t = 0
        while ( t < self.niter):

            # inputs
            X = np.array(train[features])
            y = np.array(train['label']).ravel()
            A = np.array(train[pa])
            w = np.array(train.weight).ravel()
            pred =  np.array(train[yname]).ravel()

            # search for unfairness ceritficate
            self.auditor.load_weights('model.h5')
            self.auditor.fit(X, y, sample_weight=w, batch_size=128, epochs=5, verbose=0)
            gamma = self.certificate(X, y, pred, A, w)
            print(gamma)

            # extract representation
            self.get_representation()
            PX = np.array(self.representation([X, 1]))[0]

            # estimate weights
            source = X
            target = (A == 1).astype('int32')
            mmd = MMD(source, target, lw=0.001, tol=0.001)
            mod = mmd.model
            mod.compile(loss=lambda y_true, y_pred: mmd.kera_cost(mod.inputs)(y_true, y_pred), 
                        optimizer=optimizers.SGD(lr=0.01, clipvalue=0.01), 
                        metrics=['accuracy'])
            
            mod.fit(X, target, batch_size=128, epochs=20, verbose=2)
            train['ww'] = mod.predict(X)
            train.loc[train[pa] == pg, 'weight'] = train.loc[train[pa] == pg, 'ww']
            print(train.weight.describe())
            
            t += 1

        
        test_x = np.array(test[features])
        test_a = np.array(test.attr).ravel()

        test_PX = np.array(self.representation([test_x, 1]))[0]
        test['ww'] = mod.predict(test_x)
        test.loc[test[pa] == pg, 'weight'] = test.loc[train[pa] == pg, 'ww']
        test_weights = np.array(test.weight).ravel()
        test_y = np.array(test['label']).ravel()
        pred =  np.array(test[yname]).ravel()
        gamma = self.certificate(test_x, test_y, pred, test_a, test_weights)

        print(gamma)
        print("end")

        return gamma

    
    def certificate(self, X, y, pred, A, weights):
        predicted = self.auditor.predict(X)[:, 0]
        #predicted = self.auditor.predict(X)[1].ravel()
        predicted = 2 * (predicted > 0.5).astype('int32') - 1
     
        y = 2 * y - 1
        accuracy = weights[predicted == y].sum() / weights.sum()
    
        attr = weights[A == pred].sum() / weights.sum()
        attr1 = weights[A == 1].sum() / weights.sum()
        
        return (accuracy - 1 + attr) / 4 

    def certify_iter(self, features, yname, nboot=10, balancing=None):

        results = np.zeros(nboot)
        for iter in np.arange(nboot):
            
            if balancing is None:
                gamma = self.certify(features, yname)
            elif balancing is "MMD":
                gamma = self.certify_rep(features, yname)
            elif balancing is "DNN_MMD":
                gamma = self.certify_dnn(features, yname)

            elif balancing is "KNN_MMD":
                gamma = self.certify_knn(features, yname)
            
            results[iter] = gamma

        return results[:].mean(), np.sqrt(results[:].var())


        