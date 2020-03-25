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
for i in range(3):
    for j in range(2):
        for k in range(5):
            for t in range(3):

                mm = MODELL()
                mm.lrate = 0.0006
                mm.outsize = 4
                mm.dict = {'CON' : {'list':['1','2'],
                        '1': {'FIL':32, 'KER': i, 'D_OUT': j, 'BN': False, 'INIT':'glorot_uniform', 'REG': 0.02 },
                        '2': {'FIL':k, 'KER': 8, 'D_OUT': 0.6, 'BN': False, 'INIT':t, 'REG': 0.01 },
                        #'3': {'FIL':48, 'KER': 2, 'D_OUT': 0.2, 'BN': False, 'INIT':'glorot_uniform', 'REG': 0.01 }
                       },
                          'LST' : {'list':[],
                       '1': {'FIL':128, 'SEQ': True, 'D_OUT': 0.4, 'BN': False,  'INIT': 'glorot_normal' },
                       '2': {'FIL':96,  'SEQ': True, 'D_OUT': 0.4, 'BN': False,  'INIT': 'glorot_normal'}
                      },
                          'DEN': {'list':[],
                       '1': {'FIL':98,  'D_OUT': 0.5, 'BN': False,  'INIT': 'glorot_uniform' },
                       '2': {'FIL':48,  'D_OUT': 0.5, 'BN': False,  'INIT': 'glorot_uniform'},
                       '3': {'FIL':16,  'D_OUT': 0.5, 'BN': False,  'INIT': 'glorot_uniform'}

                      }}
                lizz = {'1': ['KER','D_OUT'],
                        '2': ['FIL','INIT']}
                k1,k2,k3 = mm.SAVE_PLOTS_V2(lizz)
                print(k3)
