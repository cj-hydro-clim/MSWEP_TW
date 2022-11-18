import pandas as pd
import numpy as np
import os
import sys
from math import *
import numpy as np
from datetime import datetime
import math
import statistics

# Please revise the data_path accordingly
data_path = '/Users/cjchen/Desktop/git-repos/MSWEP_TW/MSWEP/01/02/input/'

Gauge_data = pd.read_csv(data_path+'Gauge_data.csv',header=None)
Er_data = pd.read_csv(data_path+'Satellite_data1(Er).csv',header=None)
Fr_data = pd.read_csv(data_path+'Satellite_data2(Fr).csv',header=None)
TR_data = pd.read_csv(data_path+'Reanalysis_data(TReAD).csv',header=None)

G_data = np.matrix(Gauge_data.iloc[2:1826,2:289], dtype='float64')
E_data = np.matrix(Er_data.iloc[2:1826,2:325], dtype='float64')
F_data = np.matrix(Fr_data.iloc[2:1826,2:325], dtype='float64')
T_data = np.matrix(TR_data.iloc[2:1826,2:325], dtype='float64')

for z in range(1,288): # Extract data from each station(Gauge_z)
    locals()['Gauge_'+str(z)] = np.array(G_data[0:1825,z-1])
    print()

for v in range(1,324): # Extract each grid Early run_data(Er_v)
    locals()['Er_'+str(v)] = np.array(E_data[0:1825,v-1])
    print()
    
for n in range(1,324): # Extract each grid Final run_data(Fr_n)
    locals()['Fr_'+str(n)] = np.array(F_data[0:1825,n-1])
    print()
    
for m in range(1,324): # Extract each grid TReAD_data(TR_n)
    locals()['TR_'+str(m)] = np.array(T_data[0:1825,m-1])
    print()
 
# i: grid number. a,b,c,d,e: Station number
## ========================================================================
n_D = pd.read_csv(data_path+'The nearest station (5).csv',header=None)
n_D = np.matrix(n_D.iloc[0:6,0:323], dtype='int')

for t in range(0,323): # Grab the specified station data
    ii=n_D[0,t]
    aa=n_D[1,t]
    bb=n_D[2,t]
    cc=n_D[3,t]
    dd=n_D[4,t]
    ee=n_D[5,t]
    
    locals()['Er_'+str(ii)] = np.array(locals()['Er_'+str(ii)])
    locals()['Er_'+str(ii)] = locals()['Er_'+str(ii)].flatten()
    locals()['Fr_'+str(ii)] = np.array(locals()['Fr_'+str(ii)])
    locals()['Fr_'+str(ii)] = locals()['Fr_'+str(ii)].flatten()
    locals()['TR_'+str(ii)] = np.array(locals()['TR_'+str(ii)])
    locals()['TR_'+str(ii)] = locals()['TR_'+str(ii)].flatten()

    locals()['Gauge_'+str(aa)] = np.array(locals()['Gauge_'+str(aa)])
    locals()['Gauge_'+str(aa)] = locals()['Gauge_'+str(aa)].flatten()
    locals()['Gauge_'+str(bb)] = np.array(locals()['Gauge_'+str(bb)])
    locals()['Gauge_'+str(bb)] = locals()['Gauge_'+str(bb)].flatten()
    locals()['Gauge_'+str(cc)] = np.array(locals()['Gauge_'+str(cc)])
    locals()['Gauge_'+str(cc)] = locals()['Gauge_'+str(cc)].flatten()
    locals()['Gauge_'+str(dd)] = np.array(locals()['Gauge_'+str(dd)])
    locals()['Gauge_'+str(dd)] = locals()['Gauge_'+str(dd)].flatten()
    locals()['Gauge_'+str(ee)] = np.array(locals()['Gauge_'+str(ee)])
    locals()['Gauge_'+str(ee)] = locals()['Gauge_'+str(ee)].flatten()

    E1_pccs = np.corrcoef(locals()['Er_'+str(ii)], locals()['Gauge_'+str(aa)])
    E2_pccs = np.corrcoef(locals()['Er_'+str(ii)], locals()['Gauge_'+str(bb)])
    E3_pccs = np.corrcoef(locals()['Er_'+str(ii)], locals()['Gauge_'+str(cc)])
    E4_pccs = np.corrcoef(locals()['Er_'+str(ii)], locals()['Gauge_'+str(dd)])
    E5_pccs = np.corrcoef(locals()['Er_'+str(ii)], locals()['Gauge_'+str(ee)])

    F1_pccs = np.corrcoef(locals()['Fr_'+str(ii)], locals()['Gauge_'+str(aa)])
    F2_pccs = np.corrcoef(locals()['Fr_'+str(ii)], locals()['Gauge_'+str(bb)])
    F3_pccs = np.corrcoef(locals()['Fr_'+str(ii)], locals()['Gauge_'+str(cc)])
    F4_pccs = np.corrcoef(locals()['Fr_'+str(ii)], locals()['Gauge_'+str(dd)])
    F5_pccs = np.corrcoef(locals()['Fr_'+str(ii)], locals()['Gauge_'+str(ee)])

    T1_pccs = np.corrcoef(locals()['TR_'+str(ii)], locals()['Gauge_'+str(aa)])
    T2_pccs = np.corrcoef(locals()['TR_'+str(ii)], locals()['Gauge_'+str(bb)])
    T3_pccs = np.corrcoef(locals()['TR_'+str(ii)], locals()['Gauge_'+str(cc)])
    T4_pccs = np.corrcoef(locals()['TR_'+str(ii)], locals()['Gauge_'+str(dd)])
    T5_pccs = np.corrcoef(locals()['TR_'+str(ii)], locals()['Gauge_'+str(ee)])

    locals()['E_pccs'+str(aa)] = E1_pccs [0,1]
    locals()['E_pccs'+str(bb)] = E2_pccs [0,1]
    locals()['E_pccs'+str(cc)] = E3_pccs [0,1]
    locals()['E_pccs'+str(dd)] = E4_pccs [0,1]
    locals()['E_pccs'+str(ee)] = E5_pccs [0,1]

    locals()['F_pccs'+str(aa)] = F1_pccs [0,1]
    locals()['F_pccs'+str(bb)] = F2_pccs [0,1]
    locals()['F_pccs'+str(cc)] = F3_pccs [0,1]
    locals()['F_pccs'+str(dd)] = F4_pccs [0,1]
    locals()['F_pccs'+str(ee)] = F5_pccs [0,1]

    locals()['T_pccs'+str(aa)] = T1_pccs [0,1]
    locals()['T_pccs'+str(bb)] = T2_pccs [0,1]
    locals()['T_pccs'+str(cc)] = T3_pccs [0,1]
    locals()['T_pccs'+str(dd)] = T4_pccs [0,1]
    locals()['T_pccs'+str(ee)] = T5_pccs [0,1]

# Find the median and square it
    locals()[str(ii)+'EM'] = statistics.median([locals()['E_pccs'+str(aa)], locals()['E_pccs'+str(bb)], locals()['E_pccs'+str(cc)], locals()['E_pccs'+str(dd)], locals()['E_pccs'+str(ee)]])**2
    locals()[str(ii)+'FM'] = statistics.median([locals()['F_pccs'+str(aa)], locals()['F_pccs'+str(bb)], locals()['F_pccs'+str(cc)], locals()['F_pccs'+str(dd)], locals()['F_pccs'+str(ee)]])**2
    locals()[str(ii)+'TM'] = statistics.median([locals()['T_pccs'+str(aa)], locals()['T_pccs'+str(bb)], locals()['T_pccs'+str(cc)], locals()['T_pccs'+str(dd)], locals()['T_pccs'+str(ee)]])**2

all_EM = []
for i in range(1,324): 
    all_EM.append(locals()[str(i)+'EM'])
all_EM = pd.DataFrame(all_EM)

all_FM = []
for i in range(1,324): 
    all_FM.append(locals()[str(i)+'FM'])
all_FM = pd.DataFrame(all_FM)

all_TM = []
for i in range(1,324): 
    all_TM.append(locals()[str(i)+'TM'])
all_TM = pd.DataFrame(all_TM)

# Record the weights of various products in each grid
# Please revise the paths accordingly
all_EM.to_csv('/Users/cjchen/Desktop/git-repos/MSWEP_TW/MSWEP/01/02/output/'+'IMERG_E(weight factors).csv',header=False, index=False)
all_FM.to_csv('/Users/cjchen/Desktop/git-repos/MSWEP_TW/MSWEP/01/02/output/'+'IMERG_F(weight factors).csv',header=False, index=False)
all_TM.to_csv('/Users/cjchen/Desktop/git-repos/MSWEP_TW/MSWEP/01/02/output/'+'TReAD(weight factors).csv',header=False, index=False)