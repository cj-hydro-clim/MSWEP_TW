import pandas as pd
import numpy as np

# Please revise the data_path accordingly
data_path = '/Users/cjchen/Desktop/git-repos/MSWEP_TW/MSWEP/03/01/input/'

# Load each grid weight factor for each product
Er_w = pd.read_csv(data_path+'IMERG_E(weight factors).csv',header=None)
Er_w = np.matrix(Er_w.iloc[0:], dtype='float64')

Fr_w = pd.read_csv(data_path+'IMERG_F(weight factors).csv',header=None)
Fr_w = np.matrix(Fr_w.iloc[0:], dtype='float64')

TR_w = pd.read_csv(data_path+'TReAD(weight factors).csv',header=None)
TR_w = np.matrix(TR_w.iloc[0:], dtype='float64')

TC_w = pd.read_csv(data_path+'TCCIP(weight factors).csv',header=None)
TC_w = np.matrix(TC_w.iloc[0:], dtype='float64')

## ========================================================================
# Select the weight factor of the product with the larger IMERG_E and IMERG_F of each grid
# Er
for a in range(1,324):
    if Er_w[a-1] >= Fr_w[a-1]:
        locals()['Er_w'+str(a)] = Er_w[a-1]
    else:
        locals()['Er_w'+str(a)] = 0   
Er_w = np.array(Er_w1)
for b in range(2,324):
    Er_w = np.append(Er_w,locals()['Er_w'+str(b)])    
# Fr
for a in range(1,324):
    if Fr_w[a-1] >= Er_w[a-1]:
        locals()['Fr_w'+str(a)] = Fr_w[a-1]
    else:
        locals()['Fr_w'+str(a)] = 0   
Fr_w = np.array(Fr_w1)
for b in range(2,324):
    Fr_w = np.append(Fr_w,locals()['Fr_w'+str(b)])
## ========================================================================

for c in range(1,324):
    locals()['Er_ww_'+str(c)] = Er_w[c-1] / (Er_w[c-1] + Fr_w[c-1] + TR_w[c-1] + TC_w[c-1])
    #locals()['Er_ww_'+str(c)] = locals()['Er_ww_'+str(c)][1]
    locals()['Fr_ww_'+str(c)] = Fr_w[c-1] / (Er_w[c-1] + Fr_w[c-1] + TR_w[c-1] + TC_w[c-1])
    #locals()['Fr_ww_'+str(c)] = locals()['Fr_ww_'+str(c)][1]
    locals()['TR_ww_'+str(c)] = TR_w[c-1] / (Er_w[c-1] + Fr_w[c-1] + TR_w[c-1] + TC_w[c-1])
    locals()['TC_ww_'+str(c)] = TC_w[c-1] / (Er_w[c-1] + Fr_w[c-1] + TR_w[c-1] + TC_w[c-1])

all_w = np.zeros((323, 4))
for d in range(0,323):
    all_w[d,0] = locals()['Er_ww_'+str(d+1)]
    all_w[d,1] = locals()['Fr_ww_'+str(d+1)]
    all_w[d,2] = locals()['TR_ww_'+str(d+1)]
    all_w[d,3] = locals()['TC_ww_'+str(d+1)]
    
all_w = pd.DataFrame(all_w)
# Please revise the path accordingly
all_w.to_csv('/Users/cjchen/Desktop/git-repos/MSWEP_TW/MSWEP/03/01/output/'+'All_Weights(%).csv',header=False, index=False)