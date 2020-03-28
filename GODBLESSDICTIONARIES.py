from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import pandas as pd
import sys
import types
import math
import keras
import pickle
from numba import cuda
from datetime import datetime
from hyperopt import fmin, tpe, hp, STATUS_OK, base,Trials
from scipy import misc
from statsmodels.tsa.seasonal import seasonal_decompose, STL, DecomposeResult
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace import kalman_filter
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from hyperopt import fmin, tpe, hp, STATUS_OK, base,Trials

tf.keras.backend.set_floatx('float32')
df = pd.read_excel('storage/clean.xlsx').dropna()
print('HELLOOOOOO')


class helpful:

    def preprocess(self,split):
        windowlength = self.dict['OTHERS']['1']['WINDOW_LEN']
        outsize = self.dict['OTHERS']['1']['OUT_SIZE']
        arr = np.asarray(self.data['sales'])
        vv =pd.read_csv('storage/vix.csv',sep=',')

        vix = np.array(vv['Şimdi'])
        for i in range(len(vix)):
            vix[i] = float(vix[i].replace(',','.'))

        dol =pd.read_csv('storage/dollar.csv',sep=',')
        dollars = np.array(dol['Şimdi'])
        for i in range(len(dollars)):
            dollars[i] = float(dollars[i].replace(',','.'))
            
            
        res = STL(arr,period=self.dict['OTHERS']['1']['PERIOD'],seasonal = 23 , trend = 25).fit()
        observed = res.observed
        a = np.concatenate([np.array(res.observed).reshape(res.observed.shape[0],1),np.array(res.seasonal).reshape(observed.shape[0],1),np.array(res.trend).reshape(observed.shape[0],1),np.array(res.resid).reshape(observed.shape[0],1).reshape(observed.shape[0],1),np.array(vix).reshape(observed.shape[0],1),np.array(dollars).reshape(observed.shape[0],1)],axis=1)
        dataz = np.swapaxes(np.array([res.observed,res.seasonal,res.trend,res.resid,vix,dollars]),0,1)
        train = dataz[:split]
        test = dataz[split:]
                
        MAX_window = self.dict['OTHERS']['1']['WINDOW_LEN']
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
        self.featuresize = TR_INP.shape[2]
        return TR_INP, TST_INP, TR_OUT, TST_OUT
    
    
    #PLOTS TEST OUTPUT FOR GIVEN EPOCH KEY
    def plotz(self):

        timereal = np.array([i for i in range(50)])
        timez = np.zeros((50,self.dict['OTHERS']['1']['OUT_SIZE']))
        for i in range(50):
            for j in range(self.dict['OTHERS']['1']['OUT_SIZE']):
                timez[i,j] = i + j
        real,pred,loss = self.model_test_out()
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


    def model_test_out(self,lose_bias=False):
        lossz, model_out_conc, test_out_conc = self.test_step()
        real = self.scaler.inverse_transform(test_out_conc.numpy())
        testpred = self.scaler.inverse_transform(model_out_conc.numpy())
        return real, testpred, lossz

    
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
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
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
            

            
            template = 'Sec : {} \n Epoch {} ---- Loss: {}  ----  Val_Loss: {}'
            #tf.print('Epoch', epoch, ': Time', time.time()-start, ': loss', self.loss_metric.result(), ': valid_loss', self.loss_metric_valid.result())
            self.hist.append(self.loss_metric.result())
            self.hist_valid.append(self.loss_metric_valid.result())
            self.loss_metric.reset_states()
            self.loss_metric_valid.reset_states()

class MODELL(helpful):
    def __init__(self):
        self.keylist = {}
        self.train_mean = None
        self.train_std = None
        self.train_data = None
        self.valid_data = None
        self.test_data = None
        self.epochz = None
        self.df = df
        self.initz = tf.keras.initializers.glorot_uniform(seed=None)
        self.loss_mse = tf.keras.losses.MeanAbsoluteError()
        self.transformerr = None
        self.loss_metric = tf.keras.metrics.Mean(name='loss_metric')
        self.loss_metric_valid = tf.keras.metrics.Mean(name='loss_metric_valid')
        self.loss_metric_test = tf.keras.metrics.Mean(name='loss_metric_test')
        self.checkpoints = {}
        self.firsttime = True
        self.data = pd.read_excel('storage/clean.xlsx').dropna()
        self.scaler = StandardScaler()
        self.FIRST_ITER = True
        self.optimizer = tf.keras.optimizers.Adam()
        self.keyz = list()
    #TRAIN ITERATION FOR MINI BATCH TRAIN
    #@tf.function
    def train_step(self,data):
        inp,real_out = data
        with tf.GradientTape() as lstm_tape:
            model_out = self.modell(inp,training=True)
            loss = self.loss_mse(model_out, real_out)
        
        gradients = lstm_tape.gradient(loss, self.modell.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.modell.trainable_variables))
        return self.loss_metric(loss)

    #VALID ITERATION FOR MINI BATCH TRAIN
    #@tf.function
    def valid_step(self,data_v):
        inp_train_v,real_out_v = data_v
        model_out_v = self.modell(inp_train_v,training=False)
        loss_value_v = self.loss_mse(model_out_v, real_out_v)
        return self.loss_metric_valid(loss_value_v)

    #ADD 1D-CONV BLOCK
    #input: Layer_NUM, input_TENSOR
    def CONV1D_block(self,Layer_NUM,inp):
        reg = tf.keras.regularizers.l2(self.dict['CON'][Layer_NUM]['REG'])
        x = tf.keras.layers.Conv1D(self.dict['CON'][Layer_NUM]['FIL'], kernel_initializer=self.dict['CON'][Layer_NUM]['INIT'], activity_regularizer = reg ,kernel_size=self.dict['CON'][Layer_NUM]['KER'])(inp)
        if self.dict['CON'][Layer_NUM]['BN']:
            x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.25)(x)
        x = tf.keras.layers.Dropout(self.dict['CON'][Layer_NUM]['D_OUT'])(x)
        return x
    #ADD LSTM BLOCK
    #input: Layer_NUM, input_TENSOR
    def LSTM_block(self,Layer_NUM,inp):
        x = tf.keras.layers.LSTM(self.dict['LST'][Layer_NUM]['FIL'],kernel_initializer=self.dict['LST'][Layer_NUM]['INIT'],return_sequences=self.dict['LST'][Layer_NUM]['SEQ'])(inp)
        if self.dict['LST'][Layer_NUM]['BN']:
            x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.25)(x)
        x = tf.keras.layers.Dropout(self.dict['LST'][Layer_NUM]['D_OUT'])(x)
        return x
    
    
    #ADD DENSE BLOCK
    #input: Layer_NUM, input_TENSOR    
    def DENSE_block(self,Layer_NUM,inp):
        x = tf.keras.layers.Dense(self.dict['DEN'][Layer_NUM]['FIL'],kernel_initializer=self.dict['DEN'][Layer_NUM]['INIT'])(inp)
        if self.dict['DEN'][Layer_NUM]['BN']:
            x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.25)(x)
        x = tf.keras.layers.Dropout(self.dict['DEN'][Layer_NUM]['D_OUT'])(x)
        return x
    
    
    #BUILD THE MODEL
    def model_parallel(self):
        inp = tf.keras.layers.Input(shape=(self.windowlength,self.featuresize))
        layer = 0
        flat = True
        for key in self.dict['CON']['list']:
            if layer == 0:
                x = self.CONV1D_block(key,inp)
                layer = layer + 1
            else:
                x = self.CONV1D_block(key,x)

        for key in self.dict['LST']['list']:
            if layer == 0:
                layer = layer + 1
                x = self.LSTM_block(key,inp)
            else:
                x = self.LSTM_block(key,x)

        if len(self.dict['LST']['list'])>0:
            if self.dict['LST'][self.dict['LST']['list'][-1]]['SEQ']==False:
                flat = False

        if layer != 0 and flat == True:
            x = tf.keras.layers.Flatten()(x)


        for key in self.dict['DEN']['list']:
            if layer == 0:
                layer = layer + 1
                x = self.DENSE_block(key,inp)
                x = tf.keras.layers.Flatten()(x)
            else:
                x = self.DENSE_block(key,x)

        out = tf.keras.layers.Dense(self.dict['OTHERS']['1']['OUT_SIZE'],use_bias = False)(x)
        self.modell=tf.keras.Model(inp,out)
        

    #CREATE DIR VARYING WITH THE ORDERED LAYER TYPES 
    #AND NUMBER OF LAYERS USED FOR EACH TYPE
    
    #CREATE SUBDIR OF ABOVE NAMED WITH 
    #EXPERIMENT DATE AND START TIME
    def CREATE_DIR(self):
        first_Con = True
        SAVE_DIR = 'storage/'

        for con in range(len(self.dict['CON']['list'])):
            if first_Con:
                SAVE_DIR = SAVE_DIR +'CON_' + str(self.dict['CON']['list'][con])
                first_Con = False
            else:
                SAVE_DIR = SAVE_DIR + '_' + str(self.dict['CON']['list'][con])

        first_LS = True

        for ls in range(len(self.dict['LST']['list'])):
            if first_LS:
                SAVE_DIR = SAVE_DIR + '_LS_' + str(self.dict['LST']['list'][ls])
                first_LS = False
            else:
                SAVE_DIR = SAVE_DIR + '_' + str(self.dict['LST']['list'][ls])

        first_DEN = True

        for dense in range(len(self.dict['DEN']['list'])):
            if first_DEN:
                SAVE_DIR = SAVE_DIR + '_D_' + str(self.dict['DEN']['list'][dense])
                first_LS = False
            else:
                SAVE_DIR = SAVE_DIR + '_' + str(self.dict['DEN']['list'][dense])
        try: 
            os.mkdir(SAVE_DIR)
            self.save_DIR = SAVE_DIR
        except:
            pass
        
        SAVE_DIR = SAVE_DIR + '/' +str(datetime.now())[:-10]
        SAVE_DIR = list(SAVE_DIR)
        SAVE_DIR[-6] = '--'
        SAVE_DIR = ''.join(SAVE_DIR)
        self.save_DIR = SAVE_DIR
        try:
            os.mkdir(SAVE_DIR)
        except:
            pass

    #CREATES SAVE NAME FOR BOTH PLOTS
    #AND THE KEY FOR HIST PLOT
    #key_VAR = CHANGING VARS WITH VALUES
    def CREATE_SAVE_NAME(self):
        first_con_f = True
        save_NAME = ''
        key_VAR = ''
        for Layer_TYP in list(self.VARS_EX.keys()):
            key_VAR = key_VAR + '\n'
            for kk,layer_NUM in enumerate(list(self.VARS_EX[Layer_TYP].keys())):
                if kk==0:
                    key_VAR = key_VAR + Layer_TYP  + '-' + layer_NUM + ':  '
                    save_NAME = save_NAME + '---'+  Layer_TYP[0]  + layer_NUM
                else:
                    key_VAR = '---' + key_VAR + Layer_TYP  + '-' + layer_NUM + ':  '
                    save_NAME = '---' + save_NAME + '---'+  Layer_TYP[0]  + layer_NUM                
                VALUES = '\n'
                save_VALUES = ''
                for VAR in list(self.VARS_EX[Layer_TYP][layer_NUM].keys()):
                        print(str(self.dict[Layer_TYP][layer_NUM][VAR]))
                        save_VALUES = save_VALUES  +  Layer_TYP[0]  + layer_NUM + '-' + VAR + '-' + str(self.dict[Layer_TYP][layer_NUM][VAR]) + '---'
                        VALUES = VALUES +  VAR + ' = ' +  str(self.dict[Layer_TYP][layer_NUM][VAR]) + '\t'
        key_VAR = key_VAR + VALUES
        save_NAME_PRED = save_NAME + 'PRED_' + save_VALUES
        save_NAME = save_NAME + save_VALUES
        return key_VAR, save_NAME, save_NAME_PRED
    
    #SAVE CONSTANT HYPERPARAMETERS OF EXPERIMENT AS TXT
    def WRITE_CONSTANTS(self):
        first_con_f = True
        SAVE_CON = ''
        key_CONST = ''
        for lt, Layer_TYP in enumerate(list(self.dict.keys())):
            if len(self.dict[Layer_TYP]['list']) > 0:
                if Layer_TYP != 'OTHERS':
                    for i,LAYER_NUM in enumerate(list(self.dict[Layer_TYP]['list'])):
                        if LAYER_NUM != 'list':
                            key_CONST = key_CONST[:-3] + ' \n '
                            key_CONST = key_CONST + Layer_TYP[0] + LAYER_NUM + ': '
                            for var in list(self.dict[Layer_TYP][LAYER_NUM].keys()):
                                if Layer_TYP in self.VARS_EX.keys() and LAYER_NUM in self.VARS_EX[Layer_TYP].keys():
                                    if var not in list(self.VARS_EX[Layer_TYP][LAYER_NUM]):
                                        key_CONST  = key_CONST + var + ': ' + str(self.dict[Layer_TYP][LAYER_NUM][var]) + ' -- '
                                else:
                                    key_CONST  = key_CONST + var + ': ' + str(self.dict[Layer_TYP][LAYER_NUM][var]) + ' -- '
        key_CONST = key_CONST + '\n'
        for other_KEY in self.dict['OTHERS']['1']:
            if other_KEY not in (self.VARS_EX['OTHERS']['1'].keys()):
                key_CONST = key_CONST + other_KEY + ': ' + str(self.dict['OTHERS']['1'][other_KEY]) + '\n'
        self.key_CONST = key_CONST
        save_NAME_CONST = self.save_DIR + '/CONSTANT_HyperParameters.txt'
        text_file = open(save_NAME_CONST , 'w')
        text_file.write(self.key_CONST)
        text_file.close()

    #SAVES HIST AND PRED PLOTS
    def SAVE_PLOTS(self):
        key_VAR, save_NAME, save_NAME_PRED = self.CREATE_SAVE_NAME()
        fig = plt.figure(figsize=(12,6))
        fig.suptitle(key_VAR)
        plt.plot(self.hist)
        plt.plot(self.hist_valid)
        plt.plot(np.full(shape=(np.array(self.hist).shape[0]),fill_value=0.2),'--r')
        plt.plot(np.full(shape=(np.array(self.hist).shape[0]),fill_value=0.3),'--b')
        plt.ylim((0.1,0.5))
        plt.savefig( self.save_DIR + '/' + save_NAME + '.png')
        
        fig2 = self.plotz()
        plt.savefig(self.save_DIR + '/' + save_NAME_PRED + '.png')
        plt.close('all')
        self.keyz.append(save_NAME)

    
    #SET INITIAL MODEL CHANGING PARAMETERS AS FIRST OF THEIR LIST VALUES
    def dict_UPDATE(self):
        for param in list(mm.VARS_EX.keys()):
            for sec_param in list(self.VARS_EX[param].keys()):
                for VAR in list(self.VARS_EX[param][sec_param].keys()):
                    self.dict[param][sec_param][VAR] = self.VARS_EX[param][sec_param][VAR][0]
                    
                    
    #RECURSIVE LOOP FOR TRAINING
    #RECURSIVE LOOP FOR TRAINING
    def GRID_TRAIN(self,LOOP_NUM = 0):
        if LOOP_NUM < self.num_of_rec -1 :
            for i in range(len(self.REC_LOOP_PARAMS[LOOP_NUM])):
                KEY_LIST = self.REC_LOOP_KEYS[LOOP_NUM]

                self.dict[KEY_LIST[0]][KEY_LIST[1]][KEY_LIST[2]] = self.REC_LOOP_PARAMS[LOOP_NUM][i]
                self.GRID_TRAIN(LOOP_NUM = LOOP_NUM + 1)
        else:

            for i in range(len(self.REC_LOOP_PARAMS[LOOP_NUM])):
                KEY_LIST = self.REC_LOOP_KEYS[LOOP_NUM]
                self.dict[KEY_LIST[0]][KEY_LIST[1]][KEY_LIST[2]] = self.REC_LOOP_PARAMS[LOOP_NUM][i]
                self.CREATE_MODEL()                
                
    #CREATE VARIABLES TO CREATE RECURSIVE TRAINING LOOP
    def CREATE_REC_VAR(self):
        self.num_of_rec = 0
        self.REC_LOOP_KEYS = list()
        self.REC_LOOP_PARAMS = list()
        for keys in list(self.VARS_EX.keys()):
            for LAY_NUM in list(self.VARS_EX[keys].keys()):
                for PARAM in  list(self.VARS_EX[keys][LAY_NUM].keys()):
                    self.REC_LOOP_KEYS.append([keys,LAY_NUM,PARAM])
                    self.REC_LOOP_PARAMS.append(self.VARS_EX[keys][LAY_NUM][PARAM])
                    self.num_of_rec = 1 + self.num_of_rec
        
    def CREATE_DATA(self):
        train_input, test_input, train_out, test_out = mm.preprocess(split = 216)
        self.windowbatch(train_input,train_out,test_input,test_out)
        self.valid_data = self.test_data

    def CREATE_MODEL(self):
        try:
            cuda.select_device(0)
            cuda.close()
        except:
            pass
        try:
            tf.keras.backend.clear_session()
        except:
            pass

        self.outsize = self.dict['OTHERS']['1']['OUT_SIZE']
        self.windowlength = self.dict['OTHERS']['1']['WINDOW_LEN']
        self.MAX_window = self.dict['OTHERS']['1']['WINDOW_LEN']
        self.batch =self.dict['OTHERS']['1']['BATCH_SIZE']
        self.period = self.dict['OTHERS']['1']['PERIOD']
        self.optimizer.learning_rate = self.dict['OTHERS']['1']['LR']
        self.epochz = self.dict['OTHERS']['1']['EPOCHS']

        if self.FIRST_ITER:
            self.CREATE_DATA()
            self.FIRST_ITER = False


        if ~(len(list(self.VARS_EX['OTHERS'].keys())) == 0 or list(self.VARS_EX['OTHERS'].keys()) == ['LR'] or list(self.VARS_EX['OTHERS'].keys()) == ['LR','EPOCHS'] or list(self.VARS_EX['OTHERS'].keys()) == ['EPOCHs']):
            self.CREATE_DATA()

        self.model_parallel()
        self.trainingz()
        self.SAVE_PLOTS()
        print(self.epochz)
        

    #CREATES SAVE NAME FOR BOTH PLOTS
    #AND THE KEY FOR HIST PLOT
    #key_VAR = CHANGING VARS WITH VALUES
    def CREATE_SAVE_NAME(self):
        first_con_f = True
        save_NAME = ''
        key_VAR = ''
        for qq,Layer_TYP in enumerate(list(self.VARS_EX.keys())):
            key_VAR = key_VAR + ' \n '
            if qq != 0:
                save_NAME = save_NAME + '--'
            for kk,layer_NUM in enumerate(list(self.VARS_EX[Layer_TYP].keys())):
                if Layer_TYP != 'OTHERS':
                    if kk==0:
                        key_VAR = key_VAR + Layer_TYP  + '-' + layer_NUM + ':  '
                        save_NAME = save_NAME + Layer_TYP[0]  + layer_NUM
                    else:
                        key_VAR = '--' + key_VAR + Layer_TYP  + '-' + layer_NUM + ':  '
                        save_NAME = '--' + save_NAME + '--'+  Layer_TYP[0]  + layer_NUM     

                
                save_VALUES = ''
                VALUES = ''

                for VAR in list(self.VARS_EX[Layer_TYP][layer_NUM].keys()):
                    save_VALUES = save_VALUES + '-' + VAR + ':' + str(self.dict[Layer_TYP][layer_NUM][VAR]) + '-'
                    VALUES = VALUES +  VAR + '=' +  str(self.dict[Layer_TYP][layer_NUM][VAR]) + '\t'
                    
                    
                if Layer_TYP != 'OTHERS':

                    key_VAR = key_VAR + VALUES
                    save_NAME_PRED = save_NAME + save_VALUES
                    save_NAME = save_NAME + save_VALUES
                else:
                    key_VAR = key_VAR[:-3] + VALUES
                    save_NAME_PRED = save_NAME[:-3] + save_VALUES
                    save_NAME = save_NAME[:-3] + save_VALUES

        key_VAR = key_VAR[3:]
        save_NAME = save_NAME[:-1]
        save_NAME_PRED = 'PRED-' + save_NAME_PRED[:-1]
        return key_VAR, save_NAME, save_NAME_PRED
    
            
mm = MODELL()
mm.dict = {'CON' : {'list': ['1','2'],
        '1': {'FIL':128, 'KER': 8, 'D_OUT': 0.5, 'BN': False, 'INIT':'glorot_uniform', 'REG': 0.01 },
        '2': {'FIL':48, 'KER': 8, 'D_OUT': 0.5, 'BN': False, 'INIT':'glorot_uniform', 'REG': 0.01 },
        '3': {'FIL':48, 'KER': 2, 'D_OUT': 0, 'BN': False, 'INIT':'glorot_uniform', 'REG': 0.01 }
                   },
           
          'LST' : {'list':[],
       '1': {'FIL':64, 'SEQ': True, 'D_OUT': 0, 'BN': False,  'INIT': 'glorot_uniform' },
       '2': {'FIL':24,  'SEQ': True, 'D_OUT': 0, 'BN': False,  'INIT': 'glorot_uniform'}
                  },
           
          'DEN': {'list':['1',],
       '1': {'FIL':72,  'D_OUT': 0, 'BN': False,  'INIT': 'glorot_uniform' },
       '2': {'FIL':48,  'D_OUT': 0, 'BN': False,  'INIT': 'glorot_uniform'},
       '3': {'FIL':16,  'D_OUT': 0, 'BN': False,  'INIT': 'glorot_uniform'}
                 },
          'OTHERS':{'list': ['1'],
                    '1': {'LR': 0.0003, 'EPOCHS':3000, 'WINDOW_LEN': 24, 'OUT_SIZE': 3,
                          'BATCH_SIZE' : 16, 'PERIOD': 16 }
                   }
          }

mm.VARS_EX = {'CON' :{'1': {
                            'KER': [4,8,16]
                           }
                     },
              
              'LST':{},
              
              'DEN' :{'1': {
                            'D_OUT': [0.2,0.35,0.5]
                           }
                     },
              'OTHERS':{'1':{
                             'LR': [8e-5,3e-5]
                            }
                       }
             }

mm.CREATE_DIR()
mm.WRITE_CONSTANTS()  
mm.dict_UPDATE()
mm.CREATE_REC_VAR()
mm.GRID_TRAIN()

with open('keyz.pkl', 'wb') as f:
    pickle.dump(mm.keyz, f)
