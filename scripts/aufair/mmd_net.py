from keras.layers import Dense, Dropout, Input, Activation
from keras.models import Model
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
from keras.regularizers import L1L2
from sklearn.model_selection import StratifiedKFold

import numpy as np
import tensorflow as tf

class MMD(object):

    def __init__(self, n_features, target=None, weight=None, lw=0.01, tol=0.001, conv=10**(-8)):

        self.lw = lw
        self.conv = conv
        self.tol = tol
        self.n_features = n_features
        self.target = tf.convert_to_tensor(target, dtype=tf.float32)
        self.w = tf.convert_to_tensor(weight, dtype=tf.float32)
        self.model = self.create_model()
        self.model_complete = self.create_complete_model()
        
        

    def linear_kernel(self, X, Y):
        return tf.reduce_sum( tf.multiply(X, Y ), 1, keep_dims=False)
        #return tf.multiply(X, Y)
        

    def cost_mmd(self, source, target):
        return np.linalg.norm(np.mean(source) - np.mean(target))

    def kera_cost(self, phi):

        def k_cost(y_true, y_pred):

            n1 = K.sum(y_true)
            n2 = K.sum(1 - y_true)
            
            #calculate MMD terms
            source = y_true * y_pred
            source = phi * source[:, tf.newaxis]
            target = 1 - y_true
            target = phi * target[:, tf.newaxis]
            discrepancy = 1 / n1 * K.sum(source, axis=0) - 1 / n2 * K.sum(target, axis=0)
            
            #calculate the bias MMD estimater (cannot be less than 0)
            mmd_distance = K.sum(discrepancy * discrepancy)
            return K.sqrt(mmd_distance)
        
        return k_cost

    def ent_cost(self, w):

        def e_cost(y_true, y_pred):
            return  -K.mean((w * y_true * K.log(y_pred) + w * (1 - y_true) * K.log(1 - y_pred)))
        return e_cost

    def custom_activation(self, x):
        return K.exp(x) / (1 + K.exp(x) / 10)

    def get_layer_gradient(self, model, inputs, outputs, layer=-1):
        grads = model.optimizer.get_gradients(model.total_loss, model.layers[layer].output)
        symb_inputs = (model._feed_inputs + model._feed_targets + model._feed_sample_weights)
        f = K.function(symb_inputs, grads)
        x, y, sample_weight = model._standardize_user_data(inputs, outputs)
        output_grad = f(x + y + sample_weight)
        return output_grad

    def create_model(self):
        get_custom_objects().update({'custom_activation': Activation(self.custom_activation)})
        
        # inputs
        inputs = Input(shape=(self.n_features, ), name='input')
        
        # 4 layers fully connected network
        layer1 = Dense(8, activation='relu')(inputs)
        layer2 = Dense(8, activation='relu')(layer1)
        layer3 = Dense(8, activation='relu')(layer2)
        layer4 = Dense(8, activation='relu')(layer3)
        
        representation = Dense(8, activation='relu', name='representation',
                        kernel_regularizer=L1L2(l1=0.0, l2=self.lw))(layer4)
        outputs = Dense(1, activation=self.custom_activation, name='weight')(representation)
        model = Model(inputs, outputs) 
        model.compile(loss=self.kera_cost(inputs), optimizer='adam')

        return model

    def fit(self, X, y, epochs=5, batch_size=512):
        model = self.model

        # 5 fold splits
        results = np.zeros(5)
        skf = StratifiedKFold(n_splits=5, shuffle=True)
        i = 0

        for train, test in skf.split(X, y):
            model.fit(X[train], y[train], verbose=0, epochs=epochs, batch_size=batch_size)
            scores = model.evaluate(X[test], y[test], verbose=0)
            results[i] = scores
            i += 1

        self.score = results.mean()


    def cross_validate_fit(self, X, y, epoch=[4, 5, 6], lw=[0.0001, 0.005]):

        best_lw = 0
        score = np.inf

        for l in lw:
            self.lw = l
            self.model = self.create_model()
            self.fit(X, y)
            if self.score < score:
                score = self.score
                best_lw = l

        self.lw = best_lw
        self.model = self.create_model()
        self.fit(X, y)

    def create_complete_model(self):

        inputs = Input(shape=(self.n_features,), name='input')
        
        # 4 layers fully connected representation-layer
        layer1 = Dense(8, activation='relu')(inputs)
        layer2 = Dense(8, activation='relu')(layer1)
        layer3 = Dense(8, activation='relu')(layer2)
        representation = Dense(8, activation='relu', name='representation',
                    kernel_regularizer=L1L2(l1=0.0, l2=0.001))(layer3)


        # fairness auditing
        flayer1 = Dense(8, activation='relu')(representation)
        flayer2 = Dense(8, activation='relu')(flayer1)
        fairness_out = Dense(1, activation='sigmoid', name='fairness', 
                kernel_regularizer=L1L2(l1=0.0, l2=0.05 ))(flayer2)

        # weight layersctivation=
        wlayer1 = Dense(8, activation='relu')(representation)
        wlayer2 = Dense(8, activation='relu')(wlayer1)
        weight_out = Dense(1, activation=self.custom_activation, name='weight',
                           kernel_regularizer=L1L2(l1=0.0, l2=0.01))(wlayer2)

        # complete architecture
        model = Model(inputs=inputs, outputs=[fairness_out, weight_out])

        # register model
        model.compile(loss={'fairness': self.ent_cost(self.w),
                            'weight': self.kera_cost(representation)},
                            optimizer='adam',
                            metrics={'fairness': 'accuracy'})

        return model

if __name__ == "__main__":
    import pandas as pd

    n = 5000

    data = pd.DataFrame(index=np.arange(n))
    data['x1'] = np.random.normal( size=n) 
    data['x2'] = np.random.normal(size=n) 
    data['noise'] = np.random.normal(scale=0.2, size=n)
    data['y'] =  np.exp( -0.3 * (data.x1 -data.x2) ** 3 + data['noise'] )
    data['y'] = data['y'] / (1 + data['y'])
    data['u'] = np.random.uniform(0, 1, size=len(data))
    data.loc[data.u > data.y, 'attr'] = -1
    data.loc[data.u < data.y, 'attr'] = 1
    data['attr'] = (data['attr'] + 1) /2

    # split train and test
    train = data.loc[np.random.choice(data.index, int(len(data)* 0.7), replace=True), :]
    test = data.drop(train.index)

    train0 = train.copy()

    
    X = np.array(train[['x1', 'x2']])
    A = np.array(train.attr).ravel()
    mmd_estimator = MMD(2, target=A, lw=0.01, tol=0.01)
    mmd_estimator.model.fit(X, A, epochs=5, batch_size=8, verbose=0)

    train0['weight'] = mmd_estimator.model.predict(np.array(train0[['x1', 'x2']]))
    train0['weight'] = (1- train0['weight']) / train0.weight 
    train0['w'] = (1 - train0['y']) / train0['y']
    train0.loc[train0.attr == 0, 'w'] = 1
    train0['l1'] = train0['weight'] * train0['x1'] * (train0.attr ==1).astype('int32')
    train0['l2'] = train0['weight'] * train0['x2'] * (train0.attr ==1).astype('int32')
    train0['r1'] = train0['x1'] * (train0.attr ==0).astype('int32')
    train0['r2'] = train0['x2'] * (train0.attr ==0).astype('int32')

    n1 = len(train0[train.attr ==1])
    n2 = len(train0[train.attr ==0])
    d = (train0.l1.sum() / n1 - train0.r1.sum() / n2) ** 2
    d += (train0.l2.sum() / n1 - train0.r2.sum() / n2) ** 2
    print(d)
    
    print(K.eval(mmd_estimator.kera_cost(X)(A, np.array(train0.w))))

        
    test_x = np.array(test[['x1', 'x2']])
    test_a = np.array(test.attr).ravel()
    
    test['weight'] = mmd_estimator.model.predict(test_x)
    test['weight'] = (1 - test.weight) / test.weight
    test['w'] = (1 - test['y']) / test['y']
    
    print(K.eval(mmd_estimator.kera_cost(test_x)(test_a,  np.array(test.w).ravel())) ) 
    print(K.eval(mmd_estimator.kera_cost(test_x)(test_a,  np.array( test['weight'] ).ravel())))
   
    
    



    


