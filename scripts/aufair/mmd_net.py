from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input, Activation
from keras.models import Model
from keras import backend as K
from keras_layer_normalization import LayerNormalization
from keras.utils.generic_utils import get_custom_objects
from keras.regularizers import L1L2

import numpy as np
import tensorflow as tf

class MMD(object):

    def __init__(self, n_features, lw=0.1, tol=0.001, conv=10**(-8)):

        self.lw = lw
        self.conv = conv
        self.tol = tol
        self.n_features = n_features
        self.model = self.create_model()
        

    def linear_kernel(self, X, Y):
        return tf.reduce_sum( tf.multiply(X, Y ), 1, keep_dims=False)
        #return tf.multiply(X, Y)
        

    def cost_mmd(self, source, target):
        return np.linalg.norm(np.mean(source) - np.mean(target))

    def kera_cost(self, phi):


        def k_cost(y_true, y_pred):
            n1 = K.sum(y_true, axis=0)
            n2 = K.sum(1 - y_true, axis=0)

            #calculate the 3 MMD terms
            source = y_true * y_pred 
            source = phi * source[:, tf.newaxis]
            target = (1 - y_true) 
            target = phi * target[:, tf.newaxis]
            #xx = self.linear_kernel(source, source)
            #xy = self.linear_kernel(source, target)
            #yy = self.linear_kernel(target, target)

            discrepancy = 1 / n1 *K.sum(source, axis=0) - 1/ n2 * K.sum(target, axis=0)
            
            #calculate the bias MMD estimater (cannot be less than 0)
            mmd_distance = K.sum(discrepancy * discrepancy)

            #K.sum(xx) - 2 * K.sum(xy) + K.sum(yy) 
            return K.sqrt(mmd_distance)
        
        return k_cost

    

    def ent_cost(self, a, w):

        a = tf.convert_to_tensor(a, dtype=tf.float32)

        def e_cost(y_true, y_pred):
            return -K.sum((1 - a) / 2 * (y_true * K.log(y_pred) + (1 - y_true)  * K.log(1 - y_pred))+ \
                    (a + 1) /2 * tf.exp(w) * (y_true * K.log(y_pred) + (1 - y_true)  * K.log(1 - y_pred)))
        
        return e_cost

    def custom_activation(self, x):
        return (K.exp(x) / (K.exp(x) / 5 + 1))

    def create_model(self):
        get_custom_objects().update({'custom_activation': Activation(self.custom_activation)})
        
        inputs = Input(shape=(self.n_features, ), name='input')
        
        # representation
        layer1 = Dense(8, activation='tanh')(inputs)
        #layer1 = Dropout(0.1)(layer1)
        layer2 = Dense(8, activation='relu')(layer1)
        #layer2 = Dropout(0.5)(layer2)
        layer3 = Dense(8, activation='relu')(layer2)
        #layer3 = Dropout(0.5)(layer2)
        layer4 = Dense(32, activation='relu')(layer3)
        layer4 = Dropout(0.5)(layer3)
        layer5 = Dense(32, activation='relu')(layer4)
        layer5 = Dropout(0.5)(layer4)

        # weight
        #normal_layer = LayerNormalization()(layer4)
        #weight_layer1 = Dense(4, activation='elu')(layer4)
       
        weight_out = Dense(1, activation='exponential', name='weight', kernel_regularizer=L1L2(l1=0.0, l2=0.0001))(layer3)

        self.inputs = inputs
        model = Model(inputs, weight_out) 

        model.compile(loss=self.kera_cost(inputs), optimizer='adam', metrics=[self.kera_cost(inputs)])
        return model 

    def create_complete_model(self):

        inputs = Input(shape=(self.n_features,), name='input')
        
        # representation
        layer1 = Dense(4, activation='elu')(inputs)
        layer2 = Dense(4, activation='elu')(layer1)

        # weight
        normal_layer = LayerNormalization()(layer2)
        weight_layer1 = Dense(4, activation='relu')(normal_layer)
        weight_out = Dense(1, activation='linear', name='weight')(weight_layer1)

        # fairness
        fairness_layer = Dense(4, activation='relu')(layer2)
        fairness_out = Dense(1, activation='sigmoid', name='fairness')(fairness_layer)

        # complete architecture
        model = Model(inputs=inputs, outputs=[weight_out, fairness_out]) 

        # register model
        model.compile(loss={'fairness': self.ent_cost(self.target, weight_out), 
                            'weight': self.mmd_cost(self.target, normal_layer)},
                            loss_weights = {"fairness": 1.0, "weight": 1000}, 
                            optimizer='adam',)

        return model

if __name__ == "__main__":
    import pandas as pd

    n = 5000

    data = pd.DataFrame(index=np.arange(n))
    data['attr'] = np.random.choice([-1, 1], n)
    data['x1'] = np.random.normal(size=n) 
    data['x2'] = np.random.normal(size=n) 
    data['noise'] = np.random.normal(scale=0.01, size=n)
    data['y'] =  np.exp( (data.x1 -data.x2)**3 + data['noise'] )
    data['y'] = data['y'] / (1 + data['y'])
    data['u'] = np.random.uniform(0, 1, size=len(data))
    data.loc[data.u > data.y, 'attr'] = -1
    data.loc[data.u < data.y, 'attr'] = 1
    data['attr'] = (data['attr'] + 1) /2

    # split train and test
    train = data.loc[np.random.choice(data.index, int(len(data)* 0.7), replace=True), :]
    test = data.drop(train.index)

    train0 = train.copy()

    mmd_estimator = MMD(2, lw=0.01, tol=0.01)
    X = np.array(train[['x1', 'x2']])
    A = np.array(train.attr).ravel()
    mmd_estimator.model.fit(X, A, epochs=10, batch_size=32)


    train0['weight'] = mmd_estimator.model.predict(np.array(train0[['x1', 'x2']]))
    train0['w'] = (1 - train0['y']) / train0['y']
    train0['l1'] = train0['weight'] * train0['x1'] * (train0.attr ==1).astype('int32')
    train0['l2'] = train0['weight'] * train0['x2'] * (train0.attr ==1).astype('int32')
    train0['r1'] = train0['x1'] * (train0.attr ==0).astype('int32')
    train0['r2'] = train0['x2'] * (train0.attr ==0).astype('int32')

    n1 = len(train0[train.attr ==1])
    n2 = len(train0[train.attr ==0])
    d = (train0.l1.sum() / n1 - train0.r1.sum() / n2) ** 2
    d += (train0.l2.sum() / n1 - train0.r2.sum() / n2) ** 2
    print(d)
    
    print(mmd_estimator.model.test_on_batch(X[:2, :], A[:2]))

        
    test_x = np.array(test[['x1', 'x2']])
    test_a = np.array(test.attr).ravel()
    
    test['weight'] = mmd_estimator.model.predict(test_x)
    test['w'] = (1 - test['y']) / test['y']
    w = mmd_estimator.model.predict(test_x[:100, :])
  
    print(K.eval(mmd_estimator.kera_cost(test_x[:100, :])(test_a[:100],  w.ravel())) ) 
    print(K.eval(mmd_estimator.kera_cost(test_x)(test_a,  np.array( test['weight'] ).ravel())))
   
    
    



    


