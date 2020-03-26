class MODELL(helpful):
    def __init__(self):
        self.batch = 4
        self.windowlength = 24
        self.featuresize = 6
        self.keylist = {}
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
        self.data = pd.read_excel('storage/clean.xlsx').dropna()
        self.outsize = 3
        self.MAX_window = 24
        self.scaler = StandardScaler()
        self.DICK = {}
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

    #ADD 1D-CONV BLOCK
    #input: Layer_NUM, input_TENSOR
    def CONV1D_block(self,Layer_NUM,inp):
        reg = tf.keras.regularizers.l2(self.CON[Layer_NUM]['REG'])
        x = tf.keras.layers.Conv1D(self.CON[Layer_NUM]['FIL'], kernel_initializer=self.CON[Layer_NUM]['INIT'], activity_regularizer = reg ,kernel_size=self.CON[Layer_NUM]['KER'])(inp)
        if self.CON[Layer_NUM]['BN']:
            x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.25)(x)
        x = tf.keras.layers.Dropout(self.CON[Layer_NUM]['D_OUT'])(x)
        return x
    #ADD LSTM BLOCK
    #input: Layer_NUM, input_TENSOR
    def LSTM_block(self,Layer_NUM,inp):
        x = tf.keras.layers.LSTM(self.LST[Layer_NUM]['FIL'],kernel_initializer=self.LST[Layer_NUM]['INIT'],return_sequences=self.LST[Layer_NUM]['SEQ'])(inp)
        if self.LST[Layer_NUM]['BN']:
            x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.25)(x)
        x = tf.keras.layers.Dropout(self.LST[Layer_NUM]['D_OUT'])(x)
        return x
    
    
    #ADD DENSE BLOCK
    #input: Layer_NUM, input_TENSOR    
    def DENSE_block(self,Layer_NUM,inp):
        x = tf.keras.layers.Dense(self.DEN[Layer_NUM]['FIL'],kernel_initializer=self.DEN[Layer_NUM]['INIT'])(inp)
        if self.DEN[Layer_NUM]['BN']:
            x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.25)(x)
        x = tf.keras.layers.Dropout(self.DEN[Layer_NUM]['D_OUT'])(x)
        return x
    #ADD 1D-CONV BLOCK
    #input: Layer_NUM, input_TENSOR
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
        os.mkdir(SAVE_DIR)
    
    def CREATE_SAVE_NAME(self):
        first_con_f = True
        save_NAME = ''
        key_VAR = ''
        for Layer_TYP in list(self.VAR_EX.keys()):
            key_VAR = key_VAR + '\n'
            for kk,layer_NUM in enumerate(list(self.VAR_EX[layer_TYP].keys())):
                if kk==0:
                    key_VAR = key_VAR + Layer_TYP  + '-' + layer_NUM + ':  '
                    save_NAME = save_NAME + '---'+  Layer_TYP[0]  + layer_NUM
                else:
                    key_VAR = '---' + key_VAR + Layer_TYP  + '-' + layer_NUM + ':  '
                    save_NAME = '---' + save_NAME + '---'+  Layer_TYP[0]  + layer_NUM                
                VALUES = '\n'
                save_VALUES = ''
                for VAR in list(self.VAR_EX[layer_TYP][layer_NUM].keys()):
                        save_VALUES = save_VALUES  + '---'+  Layer_TYP[0]  + layer_NUM + '-' + VAR[0] + '-' + self.VAR_EX[layer_TYP][layer_NUM][VAR]
                        VALUES = VALUES +  VAR + ' = ' + self.VAR_EX[layer_TYP][layer_NUM][VAR] + '\t'
        key_VAR = key_VAR + VALUES
        save_NAME = save_NAME + VALUES
        save_NAME_PRED = save_NAME + 'PRED_' + VALUES
        return key_VAR, save_NAME, save_NAME_PRED

    def WRITE_CONSTANTS(self):
        first_con_f = True
        SAVE_CON = ''
        key_CONST = ''
        for lt, Layer_TYP in enumerate(list(self.dict.keys())):
            if len(self.dict[Layer_TYP]['list']) > 0:
                for i,LAYER_NUM in enumerate(list(self.dict[Layer_TYP]['list'])):
                    if LAYER_NUM != 'list':
                        key_CONST = key_CONST[:-3] + ' \n '
                        key_CONST = key_CONST + Layer_TYP[0] + LAYER_NUM + ': '
                        for var in list(self.dict[Layer_TYP][LAYER_NUM].keys()):
                            if var not in list(self.VARS_EX[Layer_TYP][LAYER_NUM]):
                                key_CONST  = key_CONST + var + ': ' + str(self.dict[Layer_TYP][LAYER_NUM][var]) + ' -- '
        
        self.key_CONST = key_CONST[:-3]
        save_NAME_CONST = self.save_DIR + '/CONSTANT_HyperParameters.txt'
        text_file = open(save_NAME_CONST , 'w')
        text_file.write(self.key_CONST)
        text_file.close()

  
    def SAVE_PLOTS_V2(self):
        key_VAR, save_NAME, save_NAME_PRED = self.CREATE_SAVE_NAME()
        fig = plt.figure(figsize=(12,6))
        fig.suptitle(key_VAR)
        plt.plot(self.hist)
        plt.plot(self.hist_valid)
        plt.plot(np.full(shape=(np.array(self.hist).shape[0]),fill_value=0.2),'--r')
        plt.plot(np.full(shape=(np.array(self.hist).shape[0]),fill_value=0.3),'--b')
        plt.ylim((0.1,0.5))
        plt.savefig( self.save_DIR + '/' + save_NAME + '.png')
        
        fig2 = self.plotz(str(self.epochz-1) + '_epochs')
        plt.savefig(save_NAME_PRED + '.png')
        plt.close('all')

    

    def dict_UPDATE(self):
        for param in list(mm.VARS_EX.keys()):
            for sec_param in list(self.VARS_EX[param].keys()):
                for VAR in list(self.VARS_EX[param][sec_param].keys()):
                    self.dict[param][sec_param][VAR] = self.VARS_EX[param][sec_param][VAR][0]
                    
    def GRID_TRAIN(self,LOOP_NUM = 0):
        if LOOP_NUM < self.num_of_rec - 1:
            for i in range(len(self.REC_LOOP_PARAMS[LOOP_NUM])):
                KEY_LIST = self.REC_LOOP_KEYS[LOOP_NUM]
                
                self.dict[KEY_LIST[0]][KEY_LIST[1]][KEY_LIST[2]] = self.REC_LOOP_PARAMS[LOOP_NUM][i]
                self.GRID_TRAIN(LOOP_NUM = LOOP_NUM + 1)
        else:
            for i in range(len(self.REC_LOOP_PARAMS[LOOP_NUM])):
                KEY_LIST = self.REC_LOOP_KEYS[LOOP_NUM]
                self.dict[KEY_LIST[0]][KEY_LIST[1]][KEY_LIST[2]] = self.REC_LOOP_PARAMS[LOOP_NUM][i]
                print(self.dict)
                
    def CREATE_REC_VAR(self):
        self.REC_LOOP_KEYS = list()
        self.REC_LOOP_PARAMS = list()
        for keys in list(self.VARS_EX.keys()):
            print(keys)
            for LAY_NUM in list(self.VARS_EX[keys].keys()):
                for PARAM in  list(self.VARS_EX[keys][LAY_NUM].keys()):
                    self.REC_LOOP_KEYS.append([keys,LAY_NUM,PARAM])
                    self.REC_LOOP_PARAMS.append(self.VARS_EX[keys][LAY_NUM][PARAM])
        self.num_of_rec = len(self.REC_LOOP_KEYS)
                
                
bsize = 32
kernel = [(6,3),(8,6),(8,8),(4,4),(8,2)]
d_out = [0.6,0.8]
TF = [True,False]
dense_list= [['1'],['1','2']]
filt =[(48,36),(64,42),(84,64),(112,72),(128,96),(144,48)]
k3 = ''
try:
    del mm
except:
    pass
count = 0
mm = MODELL()

mm.dict = {'CON' : {'list':['1','2'],
        '1': {'FIL':32, 'KER': 8, 'D_OUT': 0, 'BN': False, 'INIT':'glorot_uniform', 'REG': 0.02 },
        '2': {'FIL':16, 'KER': 8, 'D_OUT': 0, 'BN': False, 'INIT':'glorot_uniform', 'REG': 0.01 },
        #'3': {'FIL':48, 'KER': 2, 'D_OUT': 0.2, 'BN': False, 'INIT':'glorot_uniform', 'REG': 0.01 }
       },
          'LST' : {'list':['1'],
       '1': {'FIL':12, 'SEQ': True, 'D_OUT': 0, 'BN': False,  'INIT': 'glorot_normal' },
       '2': {'FIL':96,  'SEQ': True, 'D_OUT': 0, 'BN': False,  'INIT': 'glorot_normal'}
      },
          'DEN': {'list':[],
       '1': {'FIL':98,  'D_OUT': 0.5, 'BN': False,  'INIT': 'glorot_uniform' },
       '2': {'FIL':48,  'D_OUT': 0.5, 'BN': False,  'INIT': 'glorot_uniform'},
       '3': {'FIL':16,  'D_OUT': 0.5, 'BN': False,  'INIT': 'glorot_uniform'}

      }}


mm.VARS_EX = {'CON' :{'1': {'KER': [8,6],
                            'D_OUT': [0.3,0.6],
                           },
                      '2': {'KER': [8,2],
                            'D_OUT': [0.5,0.8],
                           }
                     },
              'LST' :{'1': {'FIL': [12,24]
                           }
                     }
             }


mm.CREATE_DIR()
