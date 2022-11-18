import pandas as pd
import numpy as np
import os
import sys
from math import *
import numpy as np
from datetime import datetime
import math

# Please revise the data_path accordingly
data_path = '/Users/cjchen/Desktop/git-repos/MSWEP_TW/MSWEP/02/input/'

tw_G_lonlat = pd.read_csv(data_path+'The number of gauge included in each grid.csv',header=None,names=['lon','lat','num'])

lonlat = np.matrix(tw_G_lonlat.iloc[1:324,0:2], dtype='float64')
grid_st_num = np.matrix(tw_G_lonlat.iloc[1:324,2:3], dtype='float64')

# Calculate the distance between two coordinates
def np_getDistance(A , B ): # (Lat,Lon)
    ra = 6378140  # radius of equator: meter
    rb = 6356755  # radius of polar: meter
    flatten = 0.003353 # Partial rate of the earth
    # change angle to radians
    
    radLatA = np.radians(A[:,0])
    radLonA = np.radians(A[:,1])
    radLatB = np.radians(B[:,0])
    radLonB = np.radians(B[:,1])
 
    pA = np.arctan(rb / ra * np.tan(radLatA))
    pB = np.arctan(rb / ra * np.tan(radLatB))
    
    x = np.arccos( np.multiply(np.sin(pA),np.sin(pB)) + np.multiply(np.multiply(np.cos(pA),np.cos(pB)),np.cos(radLonA - radLonB)))
    c1 = np.multiply((np.sin(x) - x) , np.power((np.sin(pA) + np.sin(pB)),2)) / np.power(np.cos(x / 2),2)
    c2 = np.multiply((np.sin(x) + x) , np.power((np.sin(pA) - np.sin(pB)),2)) / np.power(np.sin(x / 2),2)
    dr = flatten / 8 * (c1 - c2)
    distance = 0.001 * ra * (x + dr)
    return distance

for c in range(1,324):

    locals()['grid_'+str(c)] = np.matrix(tw_G_lonlat.iloc[c:c+1,0:2], dtype='float64')

    locals()['disALL_'+str(c)] = np_getDistance(locals()['grid_'+str(c)],lonlat)
    
    locals()['disALL_'+str(c)][np.isnan(locals()['disALL_'+str(c)])] = 0

print()

# Calculate TCCIP weight formula (distance between grids, number of stations)
def calculate_W(D , S ): 
    Di=-(D)
    si=math.sqrt(S)
    e=math.exp(Di/25)
    F=e*si
    W=math.sqrt(F)
    return W

# Calculate TCCIP weight
for o in range(1,324):
    
    W=0
    
    for k in range(0,323):
        
        if locals()['disALL_'+str(o)][k:k+1,0:1] <= 16:
            
            locals()['w_'+str(k)] = calculate_W(locals()['disALL_'+str(o)][k:k+1,0:1] , grid_st_num[k:k+1,0:1] )
            
        else: 
            
            locals()['w_'+str(k)] = 0

        W = W + locals()['w_'+str(k)] 

    locals()['ALLw_'+str(o)] = sqrt(W)
    
all_TC = []
for i in range(1,324): 
    all_TC.append(locals()['ALLw_'+str(i)])
all_TC = pd.DataFrame(all_TC)

# Record the weights of TCCIP products in each grid
# Please revise the path accordingly
all_TC.to_csv('/Users/cjchen/Desktop/git-repos/MSWEP_TW/MSWEP/02/output/'+'TCCIP(weight factors).csv',header=False, index=False)