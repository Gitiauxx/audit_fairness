from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
from keras.models import Model
from keras import backend as K
from keras_layer_normalization import LayerNormalization

import numpy as np
import tensorflow as tf

class MMD(object):

    def __init__(self, XW, A, lw=0.1, tol=0.001, conv=10**(-8)):

        self.lw = lw
        self.conv = conv
        self.tol = tol
        self.source = XW
        self.target = A
        self.n_features = XW.shape[1]

        self.model = self.create_model()

    def linear_kernel(self, X, Y):
        return tf.reduce_sum( tf.multiply(X, Y ), 1, keep_dims=False)
        #return tf.multiply(X, Y)
        

    def cost_mmd(self, source, target):
        return np.linalg.norm(np.mean(source) - np.mean(target))

    def kera_cost(self, phi):


        def k_cost(y_true, y_pred):
            n1 = K.sum(y_true, axis=0)
            n2 = K.sum(1 - y_true)

            #calculate the 3 MMD terms
            source = tf.multiply(phi, y_true * y_pred / n1)
            target = tf.multiply(phi, (1 - y_true) * 1/ n2)
            xx = self.linear_kernel(source, source)
            xy = self.linear_kernel(source, target)
            yy = self.linear_kernel(target, target)
            
            #calculate the bias MMD estimater (cannot be less than 0)
            mmd_distance = K.sum(xx) - 2 * K.sum(xy) + K.sum(yy) 
            return mmd_distance
        
        return k_cost

    def mmd_cost(self, a, phi):

        #a = tf.convert_to_tensor(a, dtype=tf.float32)
        #n1 = K.sum( (y_true + 1) / 2)
        #n2 = K.sum( (1 - y_true) / 2)
        
        def k_cost(y_true, y_pred):

            n1 = K.sum( (y_true + 1) / 2)
            n2 = K.sum( (1 - y_true) / 2)
        

            #calculate the 3 MMD terms
            source = tf.multiply(phi, y_true * tf.exp(y_pred) * 1/n1)
            target = tf.multiply(phi, (1 - y_true) * 1/ n2)
            xx = self.linear_kernel(source, source)
            xy = self.linear_kernel(source, target)
            yy = self.linear_kernel(target, target)
            
            #calculate the bias MMD estimater (cannot be less than 0)
            mmd_distance = K.sum(xx) - 2 * K.sum(xy) + K.sum(yy) 
            return K.sqrt(mmd_distance)
        
        return k_cost

    def ent_cost(self, a, w):

        a = tf.convert_to_tensor(a, dtype=tf.float32)

        def e_cost(y_true, y_pred):
            return -K.sum((1 - a) / 2 * (y_true * K.log(y_pred) + (1 - y_true)  * K.log(1 - y_pred))+ \
                    (a + 1) /2 * tf.exp(w) * (y_true * K.log(y_pred) + (1 - y_true)  * K.log(1 - y_pred)))
        
        return e_cost

    def create_model(self):
        
        inputs = Input(shape=(self.n_features,), name='input')
        
        # representation
        layer1 = Dense(8, activation='elu')(inputs)
        layer1 = Dropout(0.2)(layer1)
        layer2 = Dense(8, activation='elu')(layer1)
        layer2 = Dropout(0.2)(layer2)
        layer3 = Dense(8, activation='elu')(layer2)
        layer3 = Dropout(0.2)(layer2)
        layer4 = Dense(4, activation='elu')(layer3)
        layer4 = Dropout(0.2)(layer3)

        # weight
        normal_layer = LayerNormalization()(layer4)
        weight_layer1 = Dense(4, activation='elu')(normal_layer)
        weight_out = Dense(1, activation='linear', name='weight')(weight_layer1)

        self.inputs = inputs
        model = Model(inputs=inputs, outputs=weight_out) 
        
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
    



    


