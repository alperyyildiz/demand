from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import pandas as pd
import sys
import types
from sklearn.preprocessing import StandardScaler
import math
import keras
from sklearn import preprocessing
import datetime
from hyperopt import fmin, tpe, hp, STATUS_OK, base,Trials
import pickle
import keras.backend as backend
from scipy import misc
from statsmodels.tsa.seasonal import seasonal_decompose, STL, DecomposeResult
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace import kalman_filter
from sklearn.preprocessing import StandardScaler
from hyperopt import fmin, tpe, hp, STATUS_OK, base,Trials
import pickle

tf.keras.backend.set_floatx('float64')
df = pd.read_excel('abc.xlsx').fillna(method='ffill')
print('HELLOOOOOO')

def test_step(modell):
    for it,data in enumerate(iter(modell.test_data),start=0):
        model_inp,test_out = data
        if it is 0:
            model_out_conc = tf.reshape(modell.modell(model_inp,training = False),shape=(1,self.outsize))
            test_out_conc = tf.reshape(test_out,shape=(1,self.outsize))
        else:

            model_out_conc = tf.concat((model_out_conc,tf.reshape(modell.modell(model_inp,training=False),shape=(1,self.outsize))), axis=0)
            test_out = tf.reshape(test_out,shape=(1,self.outsize))
            test_out_conc = tf.concat((test_out_conc,test_out),axis=0)
    return test_out_conc,model_out_conc
class helpful:
    def parallel_window(self, windowlength=[66,22,14], prediction_length=3,max_wlength=22,outcol=3):
        
        inptrain66, inptest66 , inpvalid66, outvalid66, outtrain66, outtest66 = self.Preprocessin(windowlength=windowlength[0]) 
        inptrain22, inptest22 , inpvalid22, outvalid, outtrain, outtest = self.Preprocessin(windowlength=windowlength[1]) 
        inptrain14, inptest14 , inpvalid14, outvalid14, outtrain14, outtest14 = self.Preprocessin(windowlength=windowlength[2]) 
        self.inptrain = np.concatenate([inptrain66,inptrain22,inptrain14],axis=1)
        self.inptest = np.concatenate([inptest66,inptest22,inptest14],axis=1)
        self.inpvalid = np.concatenate([inpvalid66,inpvalid22,inpvalid14],axis=1)
        
        train_data = tf.data.Dataset.from_tensor_slices((self.inptrain,outtrain))
        self.train_data = train_data.cache().batch(mm.batch).repeat(1)
        valid_data = tf.data.Dataset.from_tensor_slices((self.inpvalid,outvalid))
        self.valid_data = valid_data.cache().batch(mm.batch).repeat(1)    
        test_data = tf.data.Dataset.from_tensor_slices((self.inptest,outtest))
        self.test_data = test_data.cache().batch(1).repeat(1)
        

    def parallel_window_two(self, windowlength=[22,14], prediction_length=3,max_wlength=22,outcol=3):
        
        inptrain22, inptest22 , inpvalid22, outvalid, outtrain, outtest = self.Preprocessin(windowlength=windowlength[0]) 
        inptrain14, inptest14 , inpvalid14, outvalid14, outtrain14, outtest14 = self.Preprocessin(windowlength=windowlength[1]) 
        inptrain = np.concatenate([inptrain22,inptrain14],axis=1)
        inptest = np.concatenate([inptest22,inptest14],axis=1)
        inpvalid = np.concatenate([inpvalid22,inpvalid14],axis=1)
        
        train_data = tf.data.Dataset.from_tensor_slices((inptrain,outtrain))
        self.train_data = train_data.cache().batch(mm.batch).repeat(1)
 
        test_data = tf.data.Dataset.from_tensor_slices((inptest,outtest))
        self.test_data = test_data.cache().batch(1).repeat(1)

    def windowbatch(self,inptrain,outtrain,inptest,outtest):
    
        train_data = tf.data.Dataset.from_tensor_slices((inptrain,outtrain))
        self.train_data = train_data.cache().batch(mm.batch).repeat(1)
        test_data = tf.data.Dataset.from_tensor_slices((inptest,outtest))
        self.test_data = test_data.cache().batch(1).repeat(1)

        print('\r all data batched and ready \n')

class MODELL(helpful):
    def __init__(self):
        self.batch = 4
        self.windowlength = 24
        self.train_mean = None
        self.train_std = None
        self.train_data = None
        self.valid_data = None
        self.test_data = None
        self.epochz = None
        self.d_out = 0.4
        self.df = df
        self.initz = tf.keras.initializers.glorot_uniform(seed=None)
        self.loss_mse = tf.keras.losses.MeanAbsoluteError()
        self.transformerr = None
        self.loss_metric = tf.keras.metrics.Mean(name='loss_metric')
        self.loss_metric_valid = tf.keras.metrics.Mean(name='loss_metric_valid')
        self.loss_metric_test = tf.keras.metrics.Mean(name='loss_metric_test')
        self.checkpoints = {}
        self.firsttime = True
        self.data = pd.read_excel('abc.xlsx').fillna('ffill')
        self.outsize = 3
        self.MAX_window = 24
        self.scaler = StandardScaler()
    def inverse_transform(self):
        return self.inversetanz(self.model_test_pred,self.train_mean,self.train_std), self.inversetanz(self.model_test_out,self.train_mean,self.train_std)

    def reset_weights_and_model(self):
        self.modell.trainable_weights = self.modell_weightz
    

    @tf.function
    def train_step(self,data):
        inp,real_out = data
        with tf.GradientTape() as lstm_tape:
            model_out = self.modell(inp,training=True)
            loss = self.loss_mse(model_out, real_out)
        
        gradients = lstm_tape.gradient(loss, self.modell.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.modell.trainable_variables))
        return self.loss_metric(loss)
    
    @tf.function
    def valid_step(self,data_v):
        inp_train_v,real_out_v = data_v
        model_out_v = self.modell(inp_train_v,training=False)
        loss_value_v = self.loss_mse(model_out_v, real_out_v)
        return self.loss_metric_valid(loss_value_v)

    
    def test_step(self):
        for it,data in enumerate(iter(self.test_data),start=0):
            model_inp,test_out = data
            if it is 0:
                model_out_conc = tf.reshape(self.modell(model_inp,training = False),shape=(1,self.outsize))
                test_out_conc = tf.reshape(test_out,shape=(1,self.outsize))
                losz_ = self.loss_mse(model_out_conc, test_out_conc)

            else:
                model_out_conc = tf.concat((model_out_conc,tf.reshape(self.modell(model_inp,training=False),shape=(1,self.outsize))), axis=0)
                test_out = tf.reshape(test_out,shape=(1,self.outsize))
                test_out_conc = tf.concat((test_out_conc,test_out),axis=0)
                losz_ = self.loss_mse(model_out_conc, test_out_conc)
                
        return losz_, model_out_conc, test_out_conc
        


    def model_test_out(self,lose_bias=False):
        lossz, model_out_conc, test_out_conc = self.test_step()
        real = self.scaler.inverse_transform(test_out_conc.numpy())
        testpred = self.scaler.inverse_transform(model_out_conc.numpy())
        return real, testpred, lossz


    def trainingz(self):
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'A/logs/gradient_tape/' + current_time + '/train'
        valid_log_dir = '/logs/gradient_tape/' + current_time + '/valid'
        test_log_dir = '/logs/gradient_tape/' + current_time + '/test'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        valid_summary_writer = tf.summary.create_file_writer(valid_log_dir)

        test_summary_writer = tf.summary.create_file_writer(test_log_dir)
        self.hist = list()
        self.hist_valid = list()

        self.test_loss = list()

        for epoch in range(self.epochz):
            start = time.time()

            for data in self.train_data:
                loss_ = self.train_step(data)              
            with train_summary_writer.as_default():
                tf.summary.scalar('loss_metric', self.loss_metric.result(), step=epoch)
            for data_v in self.valid_data:
                self.valid_step(data_v)
            with valid_summary_writer.as_default():
                tf.summary.scalar('loss_metric_valid', self.loss_metric_valid.result(), step=epoch)
            
            if epoch%100==0 :
                __, predz, lloss  = self.model_test_out()
                if self.firsttime:
                    self.firsttime = False
                    self.checkpoints['testout'] = __
                else:
                    
                    key_name =  str(epoch) + '_epochs'
                    self.checkpoints[key_name] = predz
                    self.test_loss.append(lloss.numpy())
            
            template = 'Sec : {} \n Epoch {} ---- Loss: {}  ----  Val_Loss: {}'
            tf.print('Epoch', epoch, ': Time', time.time()-start, ': loss', self.loss_metric.result(), ': valid_loss', self.loss_metric_valid.result())
            self.hist.append(self.loss_metric.result())
            self.hist_valid.append(self.loss_metric_valid.result())
            self.loss_metric.reset_states()
            self.loss_metric_valid.reset_states()


    def model_parallel_copy(self):
        inp = tf.keras.layers.Input(shape=(self.windowlength,3))
        regularizer =  tf.keras.regularizers.l2(l=0.02)
        #,activity_regularizer=regularizer,
        x1 = tf.keras.layers.Conv1D(self.filter1,kernel_initializer=self.initz,activity_regularizer=regularizer,kernel_size=self.kernel1)(inp)
        if self.BN:
            x1 = tf.keras.layers.BatchNormalization()(x1)
        x1 = tf.keras.layers.LeakyReLU(alpha=0.25)(x1)
        x1 = tf.keras.layers.Dropout(self.d_out)(x1)
        x1 = tf.keras.layers.Conv1D(self.filter2,kernel_initializer=self.initz,activity_regularizer=regularizer,kernel_size=self.kernel2)(x1)
        if self.BN:
            x1 = tf.keras.layers.BatchNormalization()(x1)        
        x1 = tf.keras.layers.LeakyReLU(alpha=0.25)(x1)
        x1 = tf.keras.layers.Dropout(self.d_out)(x1)
        #x1 = tf.keras.layers.Conv1D(self.filter3,kernel_initializer=self.initz,activity_regularizer=regularizer,kernel_size=self.kernel3)(x1)
        #x1 = tf.keras.layers.BatchNormalization()(x1)
        #x1 = tf.keras.layers.LeakyReLU(alpha=0.2)(x1)
        #x1 = tf.keras.layers.Dropout(0.36)(x1)

        x1 = tf.keras.layers.Flatten()(x1)
        #x1 = tf.keras.layers.Dense(self.dense2)(x1)
        #x1 = tf.keras.layers.BatchNormalization()(x1)        

        #x1 = tf.keras.layers.LeakyReLU(alpha=0.2)(x1)
        #x1 = tf.keras.layers.Dropout(0.3)(x1)

        x1 = tf.keras.layers.Dense(self.dense1)(x1)

        x1 = tf.keras.layers.LeakyReLU(alpha=0.2)(x1)
        x1 = tf.keras.layers.Dropout(0.3)(x1)
        out = tf.keras.layers.Dense(self.outsize,use_bias = False)(x1)
        self.modell=tf.keras.Model(inp,out)
        
    def preprocess(self,period,split,windowlength):
        outsize = self.outsize
        arr = np.asarray(self.data)
        arr = np.abs(arr)
        train = arr[:split]
        test = arr[split:]
        
        lostvalues = int(period/2)
        
        MAX_window = self.MAX_window
        sclr = self.scaler.fit(train)
        train = self.scaler.transform(train)
        test = self.scaler.transform(test)

        rez = seasonal_decompose(train,period=period)
        rez_test = seasonal_decompose(test,period=period)

        observed = rez.observed[lostvalues:-lostvalues]
        trend = rez.trend[lostvalues:-lostvalues]
        season = rez.seasonal[lostvalues:-lostvalues]
        resid = rez.resid[lostvalues:-lostvalues]


        self.observed_test = rez_test.observed[lostvalues:-lostvalues]
        observed_test =  self.observed_test
        trend_test = rez_test.trend[lostvalues:-lostvalues]
        season_test = rez_test.seasonal[lostvalues:-lostvalues]
        resid_test = rez_test.resid[lostvalues:-lostvalues]
        print(observed.shape)
        outz = np.asarray([[np.array(observed)[i+k+windowlength] for i in range(outsize)] for k in range(observed.shape[0]-outsize-windowlength)])
        inp_res = np.array([[[ np.array(resid)[i+k+MAX_window-windowlength] for i in  range(windowlength)] for t in range(1)] for k in range(observed.shape[0]-outsize-windowlength)])
        inp_trd = np.array([[[ np.array(trend)[i+k+MAX_window-windowlength] for i in  range(windowlength)] for t in range(1)] for k in range(observed.shape[0]-outsize-windowlength)])
        inp_sea = np.array([[[ np.array(season)[i+k+MAX_window-windowlength] for i in  range(windowlength)] for t in range(1)] for k in range(observed.shape[0]-outsize-windowlength)])


        outtest = np.asarray([[np.array(observed_test)[i+k+windowlength] for i in range(outsize)] for k in range(observed_test.shape[0]-outsize-windowlength)])
        test_res = np.array([[[ np.array(resid_test)[i+k+MAX_window-windowlength] for i in  range(windowlength)] for t in range(1)] for k in range(observed_test.shape[0]-outsize-windowlength)])
        test_trd = np.array([[[ np.array(trend_test)[i+k+MAX_window-windowlength] for i in  range(windowlength)] for t in range(1)] for k in range(observed_test.shape[0]-outsize-windowlength)])
        test_sea = np.array([[[ np.array(season_test)[i+k+MAX_window-windowlength] for i in  range(windowlength)] for t in range(1)] for k in range(observed_test.shape[0]-outsize-windowlength)])

        
        self.pagez = observed_test.shape[0]-outsize-windowlength
        inpz = np.concatenate((inp_res,inp_trd,inp_sea),axis=1)
        inp_test = np.concatenate((test_res,test_trd,test_sea),axis=1)
        inpz = np.swapaxes(inpz,1,2)
        inp_test = np.swapaxes(inp_test,1,2)
        self.test_actual = self.scaler.inverse_transform(outtest)
        return inpz, inp_test, outz, outtest

    def plotz(self,keyz):

        timereal = np.array([i for i in range(50)])
        timez = np.zeros((50,self.outsize))
        for i in range(50):
            for j in range(self.outsize):
                timez[i,j] = i + j
        pred = self.checkpoints[keyz]
        fig = plt.figure(figsize=(12, 6))
        ax1, ax2,  = fig.subplots(2, 1, )
        ax1.ylim((0,6000))
        ax2.ylim((0,17000))
        bisi = self.test_actual.shape[0]
        for i in range(int(bisi/2)):
            ax1.plot(timez[i],pred[i])
            ax1.plot(timez[i],self.test_actual[i],color='black')

        for i in range(int(bisi/2)+1,2*int(bisi/2)+1):

            ax2.plot(timez[i],pred[i])
            ax2.plot(timez[i],self.test_actual[i],color='black')

        return fig

    
f1 = [16,32,64,84]
f2 = [16,32,48,64,96]
d1 = [32,48,64]
k1 = [2,3,4,6,8]
k2 = [2,4,6]
batch = [4,16,32]
d_out = 0.4
bnorm = True
learn = 0.0007


for fsize in f1:
    for fsize_2 in f2:
        for fsize_d in d1:
            for kern1 in k1:
                    for kern2 in k2:
                        for bsize in batch:
                            try:    
                                del mm
                            except: 
                                pass

                            mm = MODELL()
                            mm.outsize = 3

                            mm.filter1 = fsize
                            mm.filter2 = fsize_2
                            mm.kernel1 = kern1
                            mm.kernel2 = kern2
                            mm.dense1 = fsize_d
                            mm.windowlength=24
                            mm.batch = bsize
                            mm.BN = bnorm
                            print('f1: {} ---- f2: {}   d: {}   k1:  {}  k2: {}'.format(fsize,fsize_2,fsize_d,kern1,kern2))   
                            train_input, test_input, train_out, test_out = mm.preprocess(period=24,windowlength=mm.windowlength,split = 200)
                            mm.model_parallel_copy()
                            mm.optimizer = tf.keras.optimizers.Adam(learning_rate=lrate)
                            mm.windowbatch(train_input,train_out,test_input,test_out)
                            mm.valid_data = mm.test_data
                            mm.epochz = 1501
                            mm.trainingz()
                            key = 'f1: ' + str(mm.filter1) +  '  f2: ' + str(mm.filter2) + '  dense:' + str(mm.dense1) + '  \n  k1: ' + str(mm.kernel1)  
                            key = key + 'k2: ' + str(mm.kernel2) + '\n batch: ' + str(mm.batch) + '  lrate: ' + str(lrate) + 'd.out: ' +str(mm.d_out) +  '  BN: ' +str(mm.BN)
                            fig =plt.figure(figsize=(12,6))
                            fig.suptitle(key)
                            plt.plot(mm.hist)
                            plt.plot(mm.hist_valid)
                            plt.plot(np.full(shape=(np.array(mm.hist).shape[0]),fill_value=0.3),'--r')
                            plt.ylim((0.1,0.5))

                            key = str(mm.filter1) + '_' +  str(mm.filter2) + '_' + str(mm.dense1)+ '_' + str(mm.kernel1)+ '_' + str(mm.kernel2)+ '_' + str(mm.batch)+ '_' + str(lrate)+ '_' + str(mm.d_out)+ '_' + str(mm.BN)

                            dirr =  '/artifacts/' + key + '.png'
                            plt.savefig(dirr)


                            plt.clf()

                            fig = mm.plotz('1500_epochs')

                            dirr =  '/artifacts/preds_' +  key + '.png'

                            plt.savefig(dirr)

                            del fig          
