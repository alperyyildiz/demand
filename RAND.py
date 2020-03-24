  
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
df = pd.read_excel('clean.xlsx').dropna()
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
        self.featuresize = 6
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
        self.data = pd.read_excel('clean.xlsx').dropna()
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
            #tf.print('Epoch', epoch, ': Time', time.time()-start, ': loss', self.loss_metric.result(), ': valid_loss', self.loss_metric_valid.result())
            self.hist.append(self.loss_metric.result())
            self.hist_valid.append(self.loss_metric_valid.result())
            self.loss_metric.reset_states()
            self.loss_metric_valid.reset_states()

    def CONV1D_block(self,key,inp):
        reg = tf.keras.regularizers.l2(self.CON[key]['REG'])
        x = tf.keras.layers.Conv1D(self.CON[key]['FIL'], kernel_initializer=self.CON[key]['INIT'], activity_regularizer = reg ,kernel_size=self.CON[key]['KER'])(inp)
        if self.CON[key]['BN']:
            x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.25)(x)
        x = tf.keras.layers.Dropout(self.CON[key]['D_OUT'])(x)
        return x
    
    def LSTM_block(self,key,inp):
        x = tf.keras.layers.LSTM(self.LST[key]['FIL'],kernel_initializer=self.LST[key]['INIT'],return_sequences=self.LST[key]['SEQ'])(inp)
        if self.LST[key]['BN']:
            x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.25)(x)
        x = tf.keras.layers.Dropout(self.LST[key]['D_OUT'])(x)
        return x
    
    def DENSE_block(self,key,inp):
        x = tf.keras.layers.Dense(self.DEN[key]['FIL'],kernel_initializer=self.DEN[key]['INIT'])(inp)
        if self.DEN[key]['BN']:
            x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.25)(x)
        x = tf.keras.layers.Dropout(self.DEN[key]['D_OUT'])(x)
        return x
    
    def model_parallel_copy(self):
        inp = tf.keras.layers.Input(shape=(self.windowlength,self.featuresize))
        layer = 0
        for key in self.CON['list']:
            if layer == 0:
                x1 = self.CONV1D_block(key,inp)
            else:
                x1 = self.CONV1D_block(key,x1)

        for key in self.LST['list']:
            if layer == 0:
                x1 = self.LSTM_block(key,inp)
            else:
                x1 = self.LSTM_block(key,x1)
        if layer!=0:
            x = tf.keras.layers.Flatten()(x1)
        else:
            flat = True
        for key in self.DEN['list']:
            if layer == 0:
                x = self.DENSE_block(key,inp)
                x = tf.keras.layers.Flatten()(x)
            else:
                x = self.DENSE_block(key,x)

        out = tf.keras.layers.Dense(self.outsize,use_bias = False)(x)
        self.modell=tf.keras.Model(inp,out)

    def preprocess(self,period,split,windowlength):
        outsize = self.outsize
        arr = np.asarray(self.data['sales'])
        vv =pd.read_csv('vix.csv',sep=',')

        vix = np.array(vv['Şimdi'])
        for i in range(len(vix)):
            vix[i] = float(vix[i].replace(',','.'))

        dol =pd.read_csv('dollar.csv',sep=',')
        dollars = np.array(dol['Şimdi'])
        for i in range(len(dollars)):
            dollars[i] = float(dollars[i].replace(',','.'))
            
            
        res = STL(arr,period=16,seasonal = 23 , trend = 25).fit()
        observed = res.observed
        print(np.array(res.observed).shape)
        print(np.array(res.resid).shape)
        print(np.array(res.seasonal).shape)
        print(np.array(res.trend).shape)
        print(np.array(vix).shape)
        print(np.array(dollars).shape)
        a = np.concatenate([np.array(res.observed).reshape(observed.shape[0],1),np.array(res.seasonal).reshape(observed.shape[0],1),np.array(res.trend).reshape(observed.shape[0],1),np.array(res.resid).reshape(observed.shape[0],1).reshape(observed.shape[0],1),np.array(vix).reshape(observed.shape[0],1),np.array(dollars).reshape(observed.shape[0],1)],axis=1)
        dataz = np.swapaxes(np.array([res.observed,res.seasonal,res.trend,res.resid,vix,dollars]),0,1)
        train = dataz[:split]
        test = dataz[split:]
                
        MAX_window = self.MAX_window
        scaler = StandardScaler()
        sclr = scaler.fit(train)
        train =  scaler.transform(train)
        test =  scaler.transform(test)
        
        self.scaler.fit(arr[:split].reshape(-1,1))
        TR_OUT = np.asarray([[np.array(train[:,0])[i+k+windowlength] for i in range(outsize)] for k in range(split - outsize - MAX_window)])
        for feat in range(train.shape[1]):
            if feat == 0:
                TR_INP = np.array([[[ np.array(train[:,feat])[i+k+MAX_window-windowlength] for i in  range(windowlength)] for t in range(1)] for k in range(split - outsize - MAX_window)])
            else:
                TR_new = np.array([[[ np.array(train[:,feat])[i+k+MAX_window-windowlength] for i in  range(windowlength)] for t in range(1)] for k in range(split - outsize - MAX_window)])
                TR_INP = np.concatenate((TR_INP,TR_new),axis=1)

        TST_OUT = np.asarray([[np.array(test[:,0])[i+k+windowlength] for i in range(outsize)] for k in range(len(arr) - split - outsize - windowlength)])
        for feat in range(test.shape[1]):
            if feat == 0:
                TST_INP = np.array([[[ np.array(test[:,feat])[i+k+MAX_window-windowlength] for i in  range(windowlength)] for t in range(1)] for k in range(len(arr) - split - outsize - MAX_window)])
            else:
                TST_new = np.array([[[ np.array(test[:,feat])[i+k+MAX_window-windowlength] for i in  range(windowlength)] for t in range(1)] for k in range(len(arr) - split - outsize - MAX_window)])
                TST_INP = np.concatenate((TST_INP,TST_new),axis=1)

        TR_INP = np.swapaxes(TR_INP,1,2)
        TST_INP = np.swapaxes(TST_INP,1,2)
        self.pagez = test.shape[0]-outsize-windowlength
        self.test_actual = self.scaler.inverse_transform(TST_OUT)
        return TR_INP, TST_INP, TR_OUT, TST_OUT

    def plotz(self,keyz):

        timereal = np.array([i for i in range(50)])
        timez = np.zeros((50,self.outsize))
        for i in range(50):
            for j in range(self.outsize):
                timez[i,j] = i + j
        pred = self.checkpoints[keyz]
        fig = plt.figure(figsize=(12, 6))
        ax1, ax2,  = fig.subplots(2, 1, )
        bisi = self.test_actual.shape[0]
        for i in range(int(bisi/2)):
            ax1.plot(timez[i],pred[i])
            ax1.plot(timez[i],self.test_actual[i],color='black')

        for i in range(int(bisi/2)+1,2*int(bisi/2)):

            ax2.plot(timez[i],pred[i])
            ax2.plot(timez[i],self.test_actual[i],color='black')

        return fig
    
    def SAVE_PLOTS(self):
        first_Con = True
        SAVE_DIR = '/artifacts/'
        for con in range(len(self.CON['list'])):
            if first_Con:
                SAVE_DIR = SAVE_DIR +'CON_' + str(self.CON['list'][con])
                first_Con = False
            else:
                SAVE_DIR = SAVE_DIR + '_' + str(self.CON['list'][con])

        first_LS = True

        for ls in range(len(self.LST['list'])):
            if first_LS:
                SAVE_DIR = SAVE_DIR + '_LS_' + str(self.LST['list'][ls])
                first_LS = False
            else:
                SAVE_DIR = SAVE_DIR + '_' + str(self.LST['list'][ls])

        first_DEN = True

        for dense in range(len(self.DEN['list'])):
            if first_DEN:
                SAVE_DIR = SAVE_DIR + '_D_' + str(self.DEN['list'][dense])
                first_LS = False
            else:
                SAVE_DIR = SAVE_DIR + '_' + str(self.DEN['list'][dense])
        try: 
            os.mkdir(SAVE_DIR)
        except:
            pass

        first_con_f = True
        SAVE_CON = ''
        key = 'lr-' + str(self.lrate)[:7] + '_'
        key2 = 'lr-' + str(self.lrate)[:7] + '_'
        for con in self.CON['list']:
            if first_con_f:
                SAVE_CON = 'C' + con + '-' + str(self.CON[con]['FIL'])
                first_con_f = False
                key = key + 'C' + con + '_ker-' + str(self.CON[con]['KER']) +  '_DO' + '-' + str(self.CON[con]['D_OUT'])  + '_' + 'BN' + '-' + str(self.CON[con]['BN'])  + '_' + 'RG' + '-' + str(self.CON[con]['REG'])
                key2 = key2 + 'C' + con + '_ker-' + str(self.CON[con]['KER'])  +  '_DO' + '-' + str(self.CON[con]['D_OUT'])  + '_' + 'BN' + '-' + str(self.CON[con]['BN']) + '_' + 'RG' + '-' + str(self.CON[con]['REG']) + '\n'
            else:
                SAVE_CON = SAVE_CON + '_C' + con + '-' + str(self.CON[con]['FIL'])
                key = '_' + key +  '__C' + con+ '_ker-' + str(self.CON[con]['KER']) + '_' + 'DO' + '-' + str(self.CON[con]['D_OUT']) + '_BN' + '-' + str(self.CON[con]['BN'])  + '_RG' + '-' + str(self.CON[con]['REG'])
                key2 = key2 + 'C' + con+ '_ker-' + str(self.CON[con]['KER']) +  '_DO' + '-' + str(self.CON[con]['D_OUT'])  + '_BN' + '-' + str(self.CON[con]['BN']) + '_RG' + '-' + str(self.CON[con]['REG']) + '\n'

        if ~first_con_f:
            SAVE_DIR = SAVE_DIR + '/' + SAVE_CON
        try:
            os.mkdir(SAVE_DIR)
        except:
            pass  

        SAVE_LS = ''
        first_lstm_f = True
        for ls in self.LST['list']:
            if first_lstm_f:
                SAVE_LS = 'LS' + '_' + ls + '-' + str(self.LST[ls]['FIL'])
                first_lstm_f = False
                key = '_' + key +  '__LS' + ls + '_DO' + '-' + str(self.LST[ls]['D_OUT']) 
                key2 = '_' + key +  '__LS' + ls + '_DO' + '-' + str(self.LST[ls]['D_OUT']) + '\n'
            else:
                SAVE_LS = SAVE_LS + '_LS' + '_' + ls + '-' + str(self.LST[ls]['FIL'])
                key = '_' + key +  '__LS' + ls + '_DO' + '-' + str(self.LST[ls]['D_OUT']) +  '-' + str(self.LST[ls]['INIT'])
                key2 = '_' + key2 +  '__LS' + ls + '_DO' + '-' + str(self.LST[ls]['D_OUT']) + '-' + str(self.LST[ls]['INIT']) + '\n'
        if ~first_lstm_f:
            SAVE_DIR = SAVE_DIR + '/' + SAVE_LS

        try:
            os.mkdir(SAVE_DIR)
        except:
            pass    

        SAVE_DEN = ''
        first_dense_f = True
        for dense in self.DEN['list']:
            if first_dense_f:
                SAVE_DEN = 'D' + '_' + dense + '-' + str(self.DEN[dense]['FIL'])
                first_dense_f = False
                key = '_' + key +  '__DEN' + dense + '_' + 'DO' + '-' + str(self.DEN[dense]['D_OUT']) + '_BN' + '-' + str(self.DEN[dense]['BN']) 
                key2 = '_' + key2 +  '__DEN' + dense + '_' + 'DO' + '-' + str(self.DEN[dense]['D_OUT']) + '_BN' + '-' + str(self.DEN[dense]['BN']) + '\n'
            else:
                SAVE_DEN = SAVE_DEN + '_D' + '_' + dense + '-' + str(self.DEN[dense]['FIL'])
                key = '_' + key +  '__DEN' + dense + '_' + 'DO' + '-' + str(self.DEN[dense]['D_OUT']) + '_BN' + '-' + str(self.DEN[dense]['BN'])  
                key2 = '_' + key2 +  '__DEN' + dense + '_' + 'DO' + '-' + str(self.DEN[dense]['D_OUT']) + '_BN' + '-' + str(self.DEN[dense]['BN'])  + '\n'

        if ~first_dense_f:
            SAVE_DIR = SAVE_DIR + '/' + SAVE_DEN 

        try:
            os.mkdir(SAVE_DIR)
        except:
            pass    

        fig = plt.figure(figsize=(12,6))
        fig.suptitle(key2)
        plt.plot(self.hist)
        plt.plot(self.hist_valid)
        plt.plot(np.full(shape=(np.array(self.hist).shape[0]),fill_value=0.2),'--r')
        plt.plot(np.full(shape=(np.array(self.hist).shape[0]),fill_value=0.3),'--b')
        plt.ylim((0.1,0.5))
        plt.savefig( SAVE_DIR + '/' + key + '.png')
        fig2 = self.plotz(str(self.epochz-1) + '_epochs')
        plt.savefig( SAVE_DIR + '/'+ 'preds_' + key + '.png')
        plt.close('all')




bsize = 32
kernel = [(6,3),(8,6),(8,8),(4,4),(8,2)]
d_out = [0.6,0.8]
TF = [True,False]
dense_list= [['1'],['1','2']]
filt =[(48,36),(64,42),(84,64),(112,72),(128,96),(144,48)]
for fil1,fil2 in filt:
    for dout1 in d_out:
        for dlist in dense_list:
            for ker1,ker2 in kernel:
                try:
                    del mm
                except:
                    pass
                mm = MODELL()
                mm.lrate = 0.0006
                mm.outsize = 4
                mm.CON = {'list':['1','2'],
                        '1': {'FIL':fil1, 'KER': ker1, 'D_OUT': dout1, 'BN': False, 'INIT':'glorot_uniform', 'REG': 0.02 },
                        '2': {'FIL':fil2, 'KER': ker2, 'D_OUT': dout1, 'BN': False, 'INIT':'glorot_uniform', 'REG': 0.01 },
                        #'3': {'FIL':48, 'KER': 2, 'D_OUT': 0.2, 'BN': False, 'INIT':'glorot_uniform', 'REG': 0.01 }
                       }
                mm.LST = {'list':[],
                       '1': {'FIL':128, 'SEQ': True, 'D_OUT': 0.4, 'BN': False,  'INIT': 'glorot_normal' },
                       '2': {'FIL':96,  'SEQ': True, 'D_OUT': 0.4, 'BN': False,  'INIT': 'glorot_normal'}
                      }
                mm.DEN = {'list':dlist,
                       '1': {'FIL':98,  'D_OUT': 0.5, 'BN': False,  'INIT': 'glorot_uniform' },
                       '2': {'FIL':48,  'D_OUT': 0.5, 'BN': False,  'INIT': 'glorot_uniform'},
                       '3': {'FIL':16,  'D_OUT': 0.5, 'BN': False,  'INIT': 'glorot_uniform'}

                      }

                mm.windowlength=24
                mm.MAX_window = 24
                mm.batch = bsize
                train_input, test_input, train_out, test_out = mm.preprocess(period=24,windowlength=mm.windowlength,split = 216)
                mm.model_parallel_copy()
                mm.optimizer = tf.keras.optimizers.Adam(learning_rate=mm.lrate)
                mm.windowbatch(train_input,train_out,test_input,test_out)
                mm.valid_data = mm.test_data
                mm.epochz = 501
                mm.trainingz()
                mm.SAVE_PLOTS()
