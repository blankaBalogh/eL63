'''

Neural Network Model

'''

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras import layers
from metrics import *


class ML_model():
    '''
    '''
    
    def __init__(self, dic=None):

        self.in_dim = 3
        self.out_dim = 3
        self.nlays = [64, 32, 16]
        self.norms = None        
        self.model = None

        self.lrate = 1e-3
        self.epochs = 15
        self.monitor = 'val_r2_score_keras'
        self.batch_size = 128
        self.verbose = 0

        self.set_params(dic)
        

    def set_params(self, dic):
        if dic is not None:
            for key, val in zip(dic.keys(), dic.values()):
                if key in self.__dict__.keys():
                    self.__dict__[key] = val


    def build_model(self, in_dim, out_dim):
        '''
        '''
        activation = 'relu'

        self.in_dim = in_dim
        self.out_dim = out_dim
        nlays = self.nlays
        
        lrate   = self.lrate                            # learning rate
        loss_fn = tf.keras.losses.MeanSquaredError()    # loss function. Can be custom. 
        optim   = tf.keras.optimizers.Adam(lrate)       # optimizer
        
        inp_ = tf.keras.Input(shape=self.in_dim, name='X_data')
        
        x = layers.Dense(self.nlays[0], activation=activation)(inp_)
        for k in range(len(self.nlays)-1):
            x = layers.Dense(self.nlays[k], activation=activation)(x)
            
        out_ = layers.Dense(self.out_dim, name='predictions')(x)
        
        model = tf.keras.Model(inputs=inp_, outputs=out_)
        
        model.compile(loss=loss_fn, optimizer=optim, metrics=[r2_score_keras])
        model.summary()

        self.model = model


    def tendencies(self):
        '''
        '''

        def f(t, x):

            x = np.reshape(x, (-1, self.in_dim))
            
            if self.norms is not None:
                x_ = (x - self.norms[0]) / self.norms[2]
            else:
                x_ = x
            
            y_ = self.model.predict_on_batch(x_)
            
            if self.norms is not None:
                y__ = self.norms[3]*y_ + self.norms[1]
            else:
                y__ = y_
            
            return np.reshape(y__, (1, -1))
        
        return f


if __name__ == '__main__':

    print('test')
