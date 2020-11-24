import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.optimizers as ko
import tensorflow.keras.losses as kl


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

def fmse(truth, pred) :
    return np.mean((truth-pred)**2)
