# -*- coding: utf-8 -*-
"""
Intro to ML HW5 
Audrey Zhang
Andrew ID: youyouz
"""

    
import numpy as np
import sys

#%%


class NN:
    def __init__(self):
        self.alpha = None
        self.beta = None
    
    def linearForward(self, weights, vector):
        return np.dot(weights, vector)
    
    def sigmoidForward(self, vector):
        return 1/(1+np.exp(-vector)) 
    
    def softmaxForward(self, vector):
        return np.exp(vector)/sum(np.exp(vector))
    
    def crossEntropyForward(self, y, y_hat):
        return -sum(y*np.log(y_hat))
    
    def NNForward(self, x, y):
        
        # procedure NNforward:
        # a = linearForward(x, alpha)
        # z = sigmoidForward(a)
        # b = linearForward(z, beta)
        # y_hat = softmaxForward(b)
        # J = crossentropyforward(y, y_hat)
        # o = object(a, z, b, y_hat, J)
        # return intermediate quantities o

        a = self.linearForward(self.alpha, x)
        z = self.sigmoidForward(a)
        z = np.concatenate((np.array([1]), z))
        b = self.linearForward(self.beta, z)
        y_hat = self.softmaxForward(b)
        J = self.crossEntropyForward(y, y_hat)
        return (a, z, b, y_hat, J) 
    
    def crossEntropyBackward(self, y, y_hat):
        # dJ/dy_hat
        return -y/y_hat
    
    def softmaxBackward(self, y_hat, gy_hat):
        #dy_hat/db
        t1 = np.multiply(gy_hat, y_hat)
        t2 = np.dot(gy_hat, np.outer(y_hat, y_hat))
        return t1-t2
         
    def linearBackward(self, vector, weights, gradients):
        # db/dz
        # da/dx
        gvec = np.dot(gradients, weights)[1:]
        # db/dbeta
        gmat = np.outer(gradients, vector)
        return gmat, gvec
    
    def sigmoidBackward(self, z, gz):
        # dz/da
        ga = gz * z[1:] * (1-z[1:])
        return ga
    
        
    def NNBackward(self, x, y, o):
        
        # procedure NNBackward:
        # gJ = 1
        # gy_hat = crossEntropyBackward(y, y_hat, gj)
        # gb = softmaxbackward(b, y_hat, gy_hat)
        # gbeta, gz = linearBackward(z, beta, gb)
        # ga = sigmoidbackward(a, z, gz)
        # galpha, gx = linearbackward(x, alpha, ga)
        # return galpha, gbeta 

        #a, z, b, y_hat, J 
        z = o[1]
        y_hat = o[3] 
        gy_hat = self.crossEntropyBackward(y, y_hat) 
        gb = self.softmaxBackward(y_hat, gy_hat) 
        gbeta, gz = self.linearBackward(z, self.beta, gb)
        ga = self.sigmoidBackward(z, gz)
        galpha, gx = self.linearBackward(x, self.alpha, ga)
        return galpha, gbeta
    
    def train(self, x, y, val_x, val_y, epoch, hidden_units, init_flag, gamma, trainout, valout, metricsout, nclasses = 10):
        
        y_orig = y.copy()
        y_val_orig = val_y.copy()
        
        # append 1 to x for bias term 
        x = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)
        val_x = np.concatenate((np.ones((val_x.shape[0], 1)), val_x), axis=1)
        
        y_mat = np.zeros((len(y_orig), nclasses))
        for i in range(len(y_orig)):
            y_mat[i, y_orig[i]]=1 
        
        y = y_mat 
        
        val_y_mat = np.zeros((len(y_val_orig), nclasses))
        for i in range(len(y_val_orig)):
            val_y_mat[i, y_val_orig[i]]=1 
        
        val_y = val_y_mat 
        
        if init_flag == 1:
            self.alpha = np.random.uniform(low=-0.1, high=0.1, size=(hidden_units, x.shape[1]))
            self.beta = np.random.uniform(low=-0.1, high=0.1, size=(nclasses, hidden_units +1))
            
        if init_flag == 2:
            self.alpha = np.zeros((hidden_units, x.shape[1]))
            self.beta = np.zeros((nclasses, hidden_units + 1))
        
        # cross entropies
        J_train = []
        J_val = []
        
        # start training for n epochs
        for n in range(epoch):
            for i in range(x.shape[0]):
                o = self.NNForward(x[i], y[i])

                galpha, gbeta = self.NNBackward(x[i], y[i], o)
                
                # update weights
                self.alpha -= gamma*galpha
                self.beta -= gamma*gbeta

            # at each eopch, predict for train and val datasets
            # train
            y_hat_train, crossentropy = self.predict(x, y)
            J_train.append(crossentropy)
            
            # val
            y_hat_val, crossentropy = self.predict(val_x, val_y)
            J_val.append(crossentropy) 

            
        # after all epochs, write predicted labels 
        with open(trainout, 'w') as output:
            output.write('\n'.join(list(map(str, y_hat_train))))
        
        with open(valout, 'w') as output:
            output.write('\n'.join(list(map(str, y_hat_val))))
        
        # write metrics
        err_train = self.calc_error(y_hat_train, y_orig)
        err_val = self.calc_error(y_hat_val, y_val_orig)

        with open(metricsout, 'w') as output:
            for i in range(epoch):
                output.write("epoch={} crossentropy(train): {}\n".format(i+1, J_train[i]))
                output.write("epoch={} crossentropy(validation): {}\n".format(i+1, J_val[i]))
            output.write("error(train): {}\n".format(err_train))
            output.write("error(validation): {}\n".format(err_val))
        
        return J_train, J_val

    def calc_error(self, predictions, labels):
        
        err_rate = 1 - (np.count_nonzero(labels == predictions)/len(labels))
        return err_rate
    
    def predict(self, x, y):
        y_pred = [] 
        crossentropy = 0
        for i in range(len(x)):
            o = self.NNForward(x[i], y[i])
            y_pred.append(np.argmax(o[3]))
            crossentropy+=o[4]
        crossentropy = crossentropy/len(y)
        return np.array(y_pred), crossentropy
    

#%%
def main():
    train_in = sys.argv[1]
    val_in = sys.argv[2]
    train_out = sys.argv[3]
    val_out = sys.argv[4]
    metrics_out = sys.argv[5]
    num_epoch = int(sys.argv[6])
    hidden_units = int(sys.argv[7])
    init_flag = int(sys.argv[8])
    lr = float(sys.argv[9])

    train = []
    with open(train_in, 'r') as file:
        for f in file:
            line = list(map(int, f.strip('\n').split(',')))
            train.append(line)

    val = []
    with open(val_in, 'r') as file:
        for f in file:
            line = list(map(int, f.strip('\n').split(',')))
            val.append(line)
    
    train = np.array(train)
    val = np.array(val)
    
    train_y = train[:, 0]
    train_x = train[:, 1:]
    
    val_y = val[:, 0]
    val_x = val[:, 1:]   
    
    neuralnet = NN()    
    neuralnet.train(train_x, train_y, val_x, val_y, num_epoch, hidden_units, init_flag, lr, train_out, val_out, metrics_out)
    

if __name__=='__main__':
    main()




    