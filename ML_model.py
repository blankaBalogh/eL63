'''

Neural Network Model

'''

import numpy as np

import tensorflow as tf
from tensorflow import keras

import tensorflow.keras.backend as K
from tensorflow.keras import layers
import tensorflow.keras.optimizers as ko
import tensorflow.keras.losses as kl

from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler


class ML_model():
    '''
    '''

    def __init__(self, dic=None):

        self.name = ''
        
        self.in_dim = 3
        self.out_dim = 3
        self.nlays = [64, 32, 16]
        self.norms = None
        self.set_params(dic)

        self.model = None        
        self.build_model()


    def set_params(self, dic):
        if dic is not None:
            for key, val in zip(dic.keys(), dic.values()):
                if key in self.__dict__.keys():
                    self.__dict__[key] = val


    def build_model(self):
        '''
        '''
        activation = 'relu'
        
        inp_ = tf.keras.Input(shape=self.in_dim, name='X_data')
        
        x = layers.Dense(self.nlays[0], activation=activation)(inp_)
        for k in range(1, len(self.nlays)):
            x = layers.Dense(self.nlays[k], activation=activation)(x)
            
        out_ = layers.Dense(self.out_dim, name='predictions')(x)
        
        model = tf.keras.Model(inputs=inp_, outputs=out_)
        # model.summary()

        self.model = model


    def f(self):
        '''
        '''
        def func(x):
            
            ishp = x.shape
            
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
            
            return np.reshape(y__, ishp)
        
        return func


def train_ML_model(x_data, y_data, NN_model, \
        batch_size=512, learning_rate=0.001, n_epochs=10):
    
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

    # -- Model checkpoint : saving best model weights wrt "monitor" score.
    ckpt = ModelCheckpoint('weights/weights-MLP-eL63.h5', monitor='val_r2_score_keras', save_best_only=True, verbose=0, mode="max")

    def scheduler(epoch, lr):
        if epoch < 15:
            return lr
        else:
            return lr * np.exp(-0.05)

    # -- Fitting model
    loss_fn = tf.keras.losses.MeanSquaredError()        # loss function. Can be custom. 
    optim   = tf.keras.optimizers.Adam(learning_rate)   # optimizer

    def get_lr_metric(optimizer):
        def lr(y_true, y_pred):
            return optimizer._decayed_lr(tf.float32) # I use ._decayed_lr method instead of .lr
        return lr
    lr_metric = get_lr_metric(optim)

    NN_model.model.compile(loss=loss_fn, optimizer=optim, metrics=[r2_score_keras, lr_metric])

    print('training %s model...'%NN_model.name)
    history = NN_model.model.fit(x_train, y_train, epochs=n_epochs,
        batch_size=batch_size, verbose=0, validation_data=(x_test, y_test), callbacks=[ckpt, LearningRateScheduler(scheduler)])
    
    # -- Loading best ANN weights
    NN_model.model.load_weights('weights/weights-MLP-eL63.h5')



# -- Getting 1-R2 score (= Keras R2)
def r2_score_keras(y_truth, y_pred) : 
    '''
    R2-score using numpy arrays. 
    '''
    num = K.sum((y_truth - y_pred)**2)
    denom = K.sum((y_truth - K.mean(y_truth, axis=0))**2)

    return (1-num/denom)


# -- Getting loss function
def get_loss_fn(argument) :
    '''
    Return loss function. 
    '''
    if argument in ["MSE", "mse", "Mse", "MeanSquaredError"] :
        loss = kl.MeanSquaredError()
    elif argument in ["MAE", "mae", "Mae", "MeanAbsoluteError"] :
        loss = kl.MeanAbsoluteError()
    else : 
        return("You asked for optimizer : ", argument, \
                ". Available loss functions are : MSE, MAE.")
    return loss


# -- Getting optimizer
def get_optimizer(argument, lrate, **kwargs) :
    '''
    Returns an optimizer.
    '''
    if argument in ["ADAM", "adam", "Adam"] :
        optim = ko.Adam(lrate)
    elif argument in ["rmsprop", "Rmsprop", "RMSPROP"] :
        optim = ko.RMSprop(lrate)
    elif argument in ["sgd", "SGD"] :
        optim = ko.SGD(lrate)
    else :
        return("You asked for optimizer : ", argument, \
                ". Available loss functions are : MSE, MAE.")
    return optim
