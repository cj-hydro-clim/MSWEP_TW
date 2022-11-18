import pandas as pd
import numpy as np
import os
import sys
import numpy as np
# Data type must be consistent
# Grid coordinates are arranged from north to south (lon)
# i: Grid number. j: number of days

# Please revise the data_path accordingly
data_path = '/Users/cjchen/Desktop/git-repos/MSWEP_TW/MSWEP/03/02/input/'
# Load rainfall data
iE = pd.read_csv(data_path+'P_IMERG_E.csv',header=None)
iF = pd.read_csv(data_path+'P_IMERG_F.csv',header=None)
Tr = pd.read_csv(data_path+'P_TReAD.csv',header=None)
Tc = pd.read_csv(data_path+'P_TCCIP.csv',header=None)
# Load Weight (%)
w = pd.read_csv(data_path+'All_Weights(%).csv',header=None)


iE = np.matrix(iE.iloc[1:324,3:1829], dtype='float64')
iF = np.matrix(iF.iloc[1:324,3:1829], dtype='float64')
Tr = np.matrix(Tr.iloc[1:324,3:1829], dtype='float64')
Tc = np.matrix(Tc.iloc[1:324,3:1829], dtype='float64')
w = np.matrix(w.iloc[0:,0:], dtype='float64')

# Produce daily rainfall for each grid
for i in range(1,324):
    for j in range(1,1827):
    
        locals()['zp_'+str(j)] = iE[i-1,j-1]*w[i-1,0]+iF[i-1,j-1]*w[i-1,1]+Tr[i-1,j-1]*w[i-1,2]+Tc[i-1,j-1]*w[i-1,3]

    locals()['zll_p'+str(i)] = []
    for o in range(1,1827):
        locals()['zll_p'+str(i)].append(locals()['zp_'+str(o)])
    locals()['zll_p'+str(i)] = np.array(locals()['zll_p'+str(i)])
# Merge
for v in range(1,322):
    zll_ap1 = np.vstack([zll_p1,zll_p2])
    locals()['zll_ap'+str(v+1)] = np.vstack([locals()['zll_ap'+str(v)],locals()['zll_p'+str(v+2)]])

# all_Precipitation is the final result
all_Precipitation = zll_ap322
  
all_Precipitation = pd.DataFrame(all_Precipitation)
# Please revise the path accordingly
all_Precipitation.to_csv('/Users/cjchen/Desktop/git-repos/MSWEP_TW/MSWEP/03/02/output/'+'New_Data.csv',header=False, index=False)