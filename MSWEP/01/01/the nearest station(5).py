import pandas as pd
import numpy as np
import os
import sys
from math import *
import numpy as np
import math

# Please revise the data_path accordingly
data_path = '/Users/cjchen/Desktop/git-repos/MSWEP_TW/MSWEP/01/01/input/'

# Get station coordinates
Gauge_data = pd.read_csv(data_path+'Gauge_data.csv',header=None)
G_lonlat = np.transpose(np.matrix(Gauge_data.iloc[0:2,2:289], dtype='float64'))
for b in range(1,288):
    locals()['Gauge_'+str(b)] = np.matrix(G_lonlat[b-1], dtype='float64')
    print()


# Get grid coordinates
Er_data = pd.read_csv(data_path+'Satellite_data1(Er).csv',header=None)
g_lonlat = np.transpose(np.matrix(Er_data.iloc[0:2,2:325], dtype='float64'))
for d in range(1,324):
    locals()['grid_'+str(d)] = np.matrix(g_lonlat[d-1], dtype='float64')
    print()

# The formula for calculating the distance between two coordinates
def np_getDistance(A , B ):
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

# Find the numbers of the 5 nearest stations in each grid
for i in range(1,324):
    for c in range(1,288):
            
        locals()['dis_'+str(c)] = np_getDistance(locals()['grid_'+str(i)],locals()['Gauge_'+str(c)])
        
        #locals()['dis_'+str(c)][np.isnan(locals()['dis_'+str(c)])] = 0
        print()
    print()
    
    
    for x in range(1,288):
            
        locals()['dis_'+str(x)] = list(map(float, locals()['dis_'+str(x)]))
        
        locals()['dis_'+str(x)] = sum(locals()['dis_'+str(x)])
        print()
    
    all_dis = []
    for q in range(1,288):
        all_dis.append(locals()['dis_'+str(q)])
        
    all_disk = sorted(all_dis)
    all_diss = sorted(range(len(all_dis)),key = lambda k : all_dis[k])
    locals()['Grid'+str(i)+'_near_station'] = np.array([i + 1 for i in all_diss][:5]).T
    locals()['Grid'+str(i)+'_near_station'] = locals()['Grid'+str(i)+'_near_station'][np.newaxis,:]

# allGrid_near_station is the 5 stations closest to each grid from grid 1 to 323 (there are 287 stations in all)
allGrid_near_station = np.concatenate((Grid1_near_station, Grid2_near_station), axis=0)
for i in range(3,324): 
    allGrid_near_station = np.concatenate((allGrid_near_station, locals()['Grid'+str(i)+'_near_station']), axis=0)
allGrid_near_station = allGrid_near_station.T

Grid_Num = np.arange(1,324)[np.newaxis,:]
Grid_Num=Grid_Num.T

Final = np.concatenate((Grid_Num, allGrid_near_station), axis=0)

Final = pd.DataFrame(Final)
# Please revise the path accordingly
Final.to_csv('/Users/cjchen/Desktop/git-repos/MSWEP_TW/MSWEP/01/01/output/'+'The nearest station (5).csv',header=False, index=False)
