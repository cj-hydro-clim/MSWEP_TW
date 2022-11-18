# Add up to 7 adjacent grids as input at the same time. (8-1:128 combinations)
import pandas as pd
import numpy as np
import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.reset_default_graph()

# Please revise the data_path accordingly
data_path = '/Users/cjchen/Desktop/git-repos/MSWEP_TW/LSTM/input/'
data = []
name = []
corr_num = np.array([range(1,129)], dtype='float64')
corr_num = corr_num.T
dafT = pd.read_csv(data_path+'test_input_8-1.csv',header=None,names=['year','month','day','1','2','3','4','5','6','7','8','TCCIP'])
data = np.array(dafT.iloc[0:1462,2:12])
test = np.array(dafT.iloc[1462:1828:,2:11])
tccip_2019 = np.array(dafT.iloc[1462:1827:,11:12])

# Data extraction
# Traning dataset
#------------------------------------------------------------------------------------------#
# The target grid is used as a separate input for calibration
data1	=	np.array(data[0:1462,[1	,9	]])

# The target grid adds 1 surrounding grid as input for correction at the same time.
data2	=	np.array(data[0:1462,[1,2	,9 	]])
data3	=	np.array(data[0:1462,[1,3	,9 	]])
data4	=	np.array(data[0:1462,[1,4	,9 	]])
data5	=	np.array(data[0:1462,[1,5	,9 	]])
data6	=	np.array(data[0:1462,[1,6	,9 	]])
data7	=	np.array(data[0:1462,[1,7	,9 	]])
data8	=	np.array(data[0:1462,[1,8	,9 	]])

# The target grid adds 2 surrounding grid as input for correction at the same time.
data9	=	np.array(data[0:1462,[1,2,3 ,9 	]])
data10  =	np.array(data[0:1462,[1,2,4	,9 	]])
data11  =	np.array(data[0:1462,[1,2,5	,9 	]])
data12  =	np.array(data[0:1462,[1,2,6	,9 	]])
data13  =	np.array(data[0:1462,[1,2,7	,9 	]])
data14  =	np.array(data[0:1462,[1,2,8	,9 	]])
data15  =	np.array(data[0:1462,[1,3,4	,9 	]])
data16  =	np.array(data[0:1462,[1,3,5	,9 	]])
data17  =	np.array(data[0:1462,[1,3,6	,9 	]])
data18  =	np.array(data[0:1462,[1,3,7	,9 	]])
data19  =	np.array(data[0:1462,[1,3,8	,9 	]])
data20  =	np.array(data[0:1462,[1,4,5	,9 	]])
data21  =	np.array(data[0:1462,[1,4,6	,9 	]])
data22  =	np.array(data[0:1462,[1,4,7	,9 	]])
data23  =	np.array(data[0:1462,[1,4,8	,9 	]])
data24  =	np.array(data[0:1462,[1,5,6	,9 	]])
data25  =	np.array(data[0:1462,[1,5,7	,9 	]])
data26  =	np.array(data[0:1462,[1,5,8	,9 	]])
data27  =	np.array(data[0:1462,[1,6,7	,9 	]])
data28  =	np.array(data[0:1462,[1,6,8	,9 	]])
data29  =	np.array(data[0:1462,[1,7,8	,9 	]])

# The target grid adds 3 surrounding grid as input for correction at the same time.
data30	=	np.array(data[0:1462,[1,2,3,4	,9	]])
data31	=	np.array(data[0:1462,[1,2,3,5	,9	]])
data32	=	np.array(data[0:1462,[1,2,3,6	,9	]])
data33	=	np.array(data[0:1462,[1,2,3,7	,9	]])
data34	=	np.array(data[0:1462,[1,2,3,8	,9	]])
data35	=	np.array(data[0:1462,[1,2,4,5	,9	]])
data36	=	np.array(data[0:1462,[1,2,4,6	,9	]])
data37	=	np.array(data[0:1462,[1,2,4,7	,9	]])
data38	=	np.array(data[0:1462,[1,2,4,8	,9	]])
data39	=	np.array(data[0:1462,[1,2,5,6	,9	]])
data40	=	np.array(data[0:1462,[1,2,5,7	,9	]])
data41	=	np.array(data[0:1462,[1,2,5,8	,9	]])
data42	=	np.array(data[0:1462,[1,2,6,7	,9	]])
data43	=	np.array(data[0:1462,[1,2,6,8	,9	]])
data44	=	np.array(data[0:1462,[1,2,7,8	,9	]])
data45	=	np.array(data[0:1462,[1,3,4,5	,9	]])
data46	=	np.array(data[0:1462,[1,3,4,6	,9	]])
data47	=	np.array(data[0:1462,[1,3,4,7	,9	]])
data48	=	np.array(data[0:1462,[1,3,4,8	,9	]])
data49	=	np.array(data[0:1462,[1,3,5,6	,9	]])
data50	=	np.array(data[0:1462,[1,3,5,7	,9	]])
data51	=	np.array(data[0:1462,[1,3,5,8	,9	]])
data52	=	np.array(data[0:1462,[1,3,6,7	,9	]])
data53	=	np.array(data[0:1462,[1,3,6,8	,9	]])
data54	=	np.array(data[0:1462,[1,3,7,8	,9	]])
data55	=	np.array(data[0:1462,[1,4,5,6	,9	]])
data56	=	np.array(data[0:1462,[1,4,5,7	,9	]])
data57	=	np.array(data[0:1462,[1,4,5,8	,9	]])
data58	=	np.array(data[0:1462,[1,4,6,7	,9	]])
data59	=	np.array(data[0:1462,[1,4,6,8	,9	]])
data60	=	np.array(data[0:1462,[1,4,7,8	,9	]])
data61	=	np.array(data[0:1462,[1,5,6,7	,9	]])
data62	=	np.array(data[0:1462,[1,5,6,8	,9	]])
data63	=	np.array(data[0:1462,[1,5,7,8	,9	]])
data64	=	np.array(data[0:1462,[1,6,7,8	,9	]])

# The target grid adds 4 surrounding grid as input for correction at the same time.
data65	=	np.array(data[0:1462,[1,2,3,4,5 ,9	]])
data66	=	np.array(data[0:1462,[1,2,3,4,6 ,9	]])
data67	=	np.array(data[0:1462,[1,2,3,4,7 ,9	]])
data68	=	np.array(data[0:1462,[1,2,3,4,8 ,9	]])
data69	=	np.array(data[0:1462,[1,2,3,5,6	,9	]])
data70	=	np.array(data[0:1462,[1,2,3,5,7	,9	]])
data71	=	np.array(data[0:1462,[1,2,3,5,8	,9	]])
data72	=	np.array(data[0:1462,[1,2,3,6,7	,9	]])
data73	=	np.array(data[0:1462,[1,2,3,6,8	,9	]])
data74	=	np.array(data[0:1462,[1,2,3,7,8	,9	]])
data75	=	np.array(data[0:1462,[1,2,4,5,6	,9	]])
data76	=	np.array(data[0:1462,[1,2,4,5,7	,9	]])
data77	=	np.array(data[0:1462,[1,2,4,5,8	,9	]])
data78	=	np.array(data[0:1462,[1,2,4,6,7	,9	]])
data79	=	np.array(data[0:1462,[1,2,4,6,8	,9	]])
data80	=	np.array(data[0:1462,[1,2,4,7,8	,9	]])
data81	=	np.array(data[0:1462,[1,2,5,6,7	,9	]])
data82	=	np.array(data[0:1462,[1,2,5,6,8	,9	]])
data83	=	np.array(data[0:1462,[1,2,5,7,8	,9	]])
data84	=	np.array(data[0:1462,[1,2,6,7,8 ,9	]])
data85	=	np.array(data[0:1462,[1,3,4,5,6	,9	]])
data86	=	np.array(data[0:1462,[1,3,4,5,7	,9	]])
data87	=	np.array(data[0:1462,[1,3,4,5,8	,9	]])
data88	=	np.array(data[0:1462,[1,3,4,6,7	,9	]])
data89	=	np.array(data[0:1462,[1,3,4,6,8	,9	]])
data90	=	np.array(data[0:1462,[1,3,4,7,8	,9	]])
data91	=	np.array(data[0:1462,[1,3,5,6,7	,9	]])
data92	=	np.array(data[0:1462,[1,3,5,6,8	,9	]])
data93	=	np.array(data[0:1462,[1,3,5,7,8	,9	]])
data94 	=	np.array(data[0:1462,[1,3,6,7,8	,9	]])
data95 	=	np.array(data[0:1462,[1,4,5,6,7	,9	]])
data96 	=	np.array(data[0:1462,[1,4,5,6,8	,9	]])
data97 	=	np.array(data[0:1462,[1,4,5,7,8	,9	]])
data98 	=	np.array(data[0:1462,[1,4,6,7,8	,9	]])
data99 	=	np.array(data[0:1462,[1,5,6,7,8	,9	]])

# The target grid adds 5 surrounding grid as input for correction at the same time.
data100	=	np.array(data[0:1462,[1,2,3,4,5,6	,9	]])
data101	=	np.array(data[0:1462,[1,2,3,4,5,7	,9	]])
data102	=	np.array(data[0:1462,[1,2,3,4,5,8	,9	]])
data103	=	np.array(data[0:1462,[1,2,3,4,6,7	,9	]])
data104	=	np.array(data[0:1462,[1,2,3,4,6,8	,9	]])
data105	=	np.array(data[0:1462,[1,2,3,4,7,8	,9	]])
data106	=	np.array(data[0:1462,[1,2,3,5,6,7	,9	]])
data107	=	np.array(data[0:1462,[1,2,3,5,6,8	,9	]])
data108	=	np.array(data[0:1462,[1,2,3,5,7,8	,9	]])
data109	=	np.array(data[0:1462,[1,2,3,6,7,8	,9	]])
data110	=	np.array(data[0:1462,[1,2,4,5,6,7	,9	]])
data111	=	np.array(data[0:1462,[1,2,4,5,6,8	,9	]])
data112	=	np.array(data[0:1462,[1,2,4,5,7,8	,9	]])
data113	=	np.array(data[0:1462,[1,2,4,6,7,8	,9	]])
data114	=	np.array(data[0:1462,[1,2,5,6,7,8	,9	]])
data115	=	np.array(data[0:1462,[1,3,4,5,6,7	,9	]])
data116	=	np.array(data[0:1462,[1,3,4,5,6,8	,9	]])
data117	=	np.array(data[0:1462,[1,3,4,5,7,8	,9	]])
data118	=	np.array(data[0:1462,[1,3,4,6,7,8	,9	]])
data119	=	np.array(data[0:1462,[1,3,5,6,7,8	,9	]])
data120	=	np.array(data[0:1462,[1,4,5,6,7,8	,9	]])

# The target grid adds 6 surrounding grid as input for correction at the same time.
data121	=	np.array(data[0:1462,[5,2,3,4,5,6,7	,9 ]])
data122	=	np.array(data[0:1462,[5,2,3,4,5,6,8	,9 ]])
data123	=	np.array(data[0:1462,[5,2,3,4,5,7,8	,9 ]])
data124	=	np.array(data[0:1462,[5,2,3,4,6,7,8	,9 ]])
data125	=	np.array(data[0:1462,[5,2,3,5,6,7,8	,9 ]])
data126	=	np.array(data[0:1462,[5,2,4,5,6,7,8	,9 ]])
data127	=	np.array(data[0:1462,[5,3,4,5,6,7,8 ,9 ]])

# The target grid adds 7 surrounding grid as input for correction at the same time.
data128	=	np.array(data[0:1462,[5,2,3,4,5,6,7,8 ,9 ]])
#------------------------------------------------------------------------------------------#

# Validation dataset
#------------------------------------------------------------------------------------------#
test1	=	np.array(test[0:366,[1		]])

test2	=	np.array(test[0:366,[1,2	 	]])
test3	=	np.array(test[0:366,[1,3	 	]])
test4	=	np.array(test[0:366,[1,4	 	]])
test5	=	np.array(test[0:366,[1,5	 	]])
test6	=	np.array(test[0:366,[1,6	 	]])
test7	=	np.array(test[0:366,[1,7	 	]])
test8	=	np.array(test[0:366,[1,8	 	]])

test9	=	np.array(test[0:366,[1,2,3   	]])
test10  =	np.array(test[0:366,[1,2,4	 	]])
test11  =	np.array(test[0:366,[1,2,5	 	]])
test12  =	np.array(test[0:366,[1,2,6	 	]])
test13  =	np.array(test[0:366,[1,2,7	 	]])
test14  =	np.array(test[0:366,[1,2,8	 	]])
test15  =	np.array(test[0:366,[1,3,4	 	]])
test16  =	np.array(test[0:366,[1,3,5	 	]])
test17  =	np.array(test[0:366,[1,3,6	 	]])
test18  =	np.array(test[0:366,[1,3,7	 	]])
test19  =	np.array(test[0:366,[1,3,8	 	]])
test20  =	np.array(test[0:366,[1,4,5	 	]])
test21  =	np.array(test[0:366,[1,4,6	 	]])
test22  =	np.array(test[0:366,[1,4,7	 	]])
test23  =	np.array(test[0:366,[1,4,8	 	]])
test24  =	np.array(test[0:366,[1,5,6	 	]])
test25  =	np.array(test[0:366,[1,5,7	 	]])
test26  =	np.array(test[0:366,[1,5,8	 	]])
test27  =	np.array(test[0:366,[1,6,7	 	]])
test28  =	np.array(test[0:366,[1,6,8	 	]])
test29  =	np.array(test[0:366,[1,7,8	 	]])

test30	=	np.array(test[0:366,[1,2,3,4		]])
test31	=	np.array(test[0:366,[1,2,3,5		]])
test32	=	np.array(test[0:366,[1,2,3,6		]])
test33	=	np.array(test[0:366,[1,2,3,7	 	]])
test34	=	np.array(test[0:366,[1,2,3,8	 	]])
test35	=	np.array(test[0:366,[1,2,4,5		]])
test36	=	np.array(test[0:366,[1,2,4,6		]])
test37	=	np.array(test[0:366,[1,2,4,7		]])
test38	=	np.array(test[0:366,[1,2,4,8		]])
test39	=	np.array(test[0:366,[1,2,5,6		]])
test40	=	np.array(test[0:366,[1,2,5,7		]])
test41	=	np.array(test[0:366,[1,2,5,8		]])
test42	=	np.array(test[0:366,[1,2,6,7		]])
test43	=	np.array(test[0:366,[1,2,6,8		]])
test44	=	np.array(test[0:366,[1,2,7,8		]])
test45	=	np.array(test[0:366,[1,3,4,5		]])
test46	=	np.array(test[0:366,[1,3,4,6		]])
test47	=	np.array(test[0:366,[1,3,4,7		]])
test48	=	np.array(test[0:366,[1,3,4,8		]])
test49	=	np.array(test[0:366,[1,3,5,6		]])
test50	=	np.array(test[0:366,[1,3,5,7		]])
test51	=	np.array(test[0:366,[1,3,5,8		]])
test52	=	np.array(test[0:366,[1,3,6,7		]])
test53	=	np.array(test[0:366,[1,3,6,8		]])
test54	=	np.array(test[0:366,[1,3,7,8		]])
test55	=	np.array(test[0:366,[1,4,5,6		]])
test56	=	np.array(test[0:366,[1,4,5,7		]])
test57	=	np.array(test[0:366,[1,4,5,8		]])
test58	=	np.array(test[0:366,[1,4,6,7		]])
test59	=	np.array(test[0:366,[1,4,6,8		]])
test60	=	np.array(test[0:366,[1,4,7,8		]])
test61	=	np.array(test[0:366,[1,5,6,7		]])
test62	=	np.array(test[0:366,[1,5,6,8		]])
test63	=	np.array(test[0:366,[1,5,7,8		]])
test64	=	np.array(test[0:366,[1,6,7,8		]])

test65	=	np.array(test[0:366,[1,2,3,4,5  	]])
test66	=	np.array(test[0:366,[1,2,3,4,6  	]])
test67	=	np.array(test[0:366,[1,2,3,4,7  	]])
test68	=	np.array(test[0:366,[1,2,3,4,8  	]])
test69	=	np.array(test[0:366,[1,2,3,5,6		]])
test70	=	np.array(test[0:366,[1,2,3,5,7		]])
test71	=	np.array(test[0:366,[1,2,3,5,8		]])
test72	=	np.array(test[0:366,[1,2,3,6,7		]])
test73	=	np.array(test[0:366,[1,2,3,6,8		]])
test74	=	np.array(test[0:366,[1,2,3,7,8		]])
test75	=	np.array(test[0:366,[1,2,4,5,6		]])
test76	=	np.array(test[0:366,[1,2,4,5,7		]])
test77	=	np.array(test[0:366,[1,2,4,5,8		]])
test78	=	np.array(test[0:366,[1,2,4,6,7		]])
test79	=	np.array(test[0:366,[1,2,4,6,8		]])
test80	=	np.array(test[0:366,[1,2,4,7,8		]])
test81	=	np.array(test[0:366,[1,2,5,6,7		]])
test82	=	np.array(test[0:366,[1,2,5,6,8		]])
test83	=	np.array(test[0:366,[1,2,5,7,8		]])
test84	=	np.array(test[0:366,[1,2,6,7,8  	]])
test85	=	np.array(test[0:366,[1,3,4,5,6		]])
test86	=	np.array(test[0:366,[1,3,4,5,7		]])
test87	=	np.array(test[0:366,[1,3,4,5,8		]])
test88	=	np.array(test[0:366,[1,3,4,6,7		]])
test89	=	np.array(test[0:366,[1,3,4,6,8		]])
test90	=	np.array(test[0:366,[1,3,4,7,8		]])
test91	=	np.array(test[0:366,[1,3,5,6,7		]])
test92	=	np.array(test[0:366,[1,3,5,6,8		]])
test93	=	np.array(test[0:366,[1,3,5,7,8		]])
test94 	=	np.array(test[0:366,[1,3,6,7,8		]])
test95 	=	np.array(test[0:366,[1,4,5,6,7		]])
test96 	=	np.array(test[0:366,[1,4,5,6,8		]])
test97 	=	np.array(test[0:366,[1,4,5,7,8		]])
test98 	=	np.array(test[0:366,[1,4,6,7,8		]])
test99 	=	np.array(test[0:366,[1,5,6,7,8		]])

test100	=	np.array(test[0:366,[1,2,3,4,5,6		]])
test101	=	np.array(test[0:366,[1,2,3,4,5,7		]])
test102	=	np.array(test[0:366,[1,2,3,4,5,8		]])
test103	=	np.array(test[0:366,[1,2,3,4,6,7		]])
test104	=	np.array(test[0:366,[1,2,3,4,6,8		]])
test105	=	np.array(test[0:366,[1,2,3,4,7,8		]])
test106	=	np.array(test[0:366,[1,2,3,5,6,7		]])
test107	=	np.array(test[0:366,[1,2,3,5,6,8		]])
test108	=	np.array(test[0:366,[1,2,3,5,7,8		]])
test109	=	np.array(test[0:366,[1,2,3,6,7,8		]])
test110	=	np.array(test[0:366,[1,2,4,5,6,7		]])
test111	=	np.array(test[0:366,[1,2,4,5,6,8		]])
test112	=	np.array(test[0:366,[1,2,4,5,7,8		]])
test113	=	np.array(test[0:366,[1,2,4,6,7,8		]])
test114	=	np.array(test[0:366,[1,2,5,6,7,8		]])
test115	=	np.array(test[0:366,[1,3,4,5,6,7		]])
test116	=	np.array(test[0:366,[1,3,4,5,6,8		]])
test117	=	np.array(test[0:366,[1,3,4,5,7,8		]])
test118	=	np.array(test[0:366,[1,3,4,6,7,8		]])
test119	=	np.array(test[0:366,[1,3,5,6,7,8		]])
test120	=	np.array(test[0:366,[1,4,5,6,7,8		]])

test121	=	np.array(test[0:366,[1,2,3,4,5,6,7	 ]])
test122	=	np.array(test[0:366,[1,2,3,4,5,6,8	 ]])
test123	=	np.array(test[0:366,[1,2,3,4,5,7,8	 ]])
test124	=	np.array(test[0:366,[1,2,3,4,6,7,8	 ]])
test125	=	np.array(test[0:366,[1,2,3,5,6,7,8	 ]])
test126	=	np.array(test[0:366,[1,2,4,5,6,7,8	 ]])
test127	=	np.array(test[0:366,[1,3,4,5,6,7,8   ]])

test128	=	np.array(test[0:366,[1,2,3,4,5,6,7,8  ]])
#------------------------------------------------------------------------------------------#

# Variable k is the file number 1~128.
# The variable j is entered and other parameters. If for file k=1,j=1. k=2~8,j=2. etc.
#    1    = 1 ( 1  )
# 2  ~  8 = 2 ( 7  )
# 9  ~ 29 = 3 ( 21 )
# 30 ~ 64 = 4 ( 35 )
# 65 ~ 99 = 5 ( 35 )
# 100~120 = 6 ( 21 )
# 121~127 = 7 ( 7  )
#   128   = 8 ( 1  )
for k in range(1,129):
    if k == 1 :
         j = 1;
    elif 2   <= k <=   8:
         j = 2;
    elif 9   <= k <=  29:
         j = 3;
    elif 30  <= k <=  64:
         j = 4;
    elif 65  <= k <=  99:
         j = 5;
    elif 100 <= k <= 120:
         j = 6;
    elif 121 <= k <= 127:
         j = 7;
    elif k == 128:
         j = 8;
    
    
# Create Convlstm model
    data = eval('data'+str(k))
    test = eval('test'+str(k))

    tf.reset_default_graph()
    tf.set_random_seed(718)
    rnn_unit = 10    # Hidden Layer
    input_size = j   # Input Layer Dimension
    output_size = 1  # Output Layer Dimension
    lr = 0.0006      # learning rate
    epochs = 500     # epochs

# Define the weights and bias of input and output layers
    weights = {
        'in': tf.Variable(tf.random.normal([input_size, rnn_unit])),
        'out': tf.Variable(tf.random.normal([rnn_unit, 1]))
        }
    biases = {
        'in': tf.Variable(tf.constant(0.1, shape=[rnn_unit, ])),
        'out': tf.Variable(tf.constant(0.1, shape=[1, ]))
        }

    def lstm(X):
        batch_size = tf.shape(X)[0]
        time_step = tf.shape(X)[1]
        w_in = weights['in']
        b_in = biases['in']
        input = tf.reshape(X, [-1, input_size])
        input_rnn = tf.matmul(input, w_in)+b_in
        input_rnn = tf.reshape(input_rnn, [-1, time_step, rnn_unit])
        cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
        init_state = cell.zero_state(batch_size, dtype=tf.float32)
        output_rnn, final_states = tf.nn.dynamic_rnn(cell, input_rnn, initial_state=init_state, dtype=tf.float32)
        output = tf.reshape(output_rnn, [-1, rnn_unit])
        w_out = weights['out']
        b_out = biases['out']
        pred = tf.matmul(output, w_out)+b_out
        return pred, final_states

    def get_train_data(batch_size=60, time_step=20,train_begin=0, train_end=len(data)):
        batch_index = []
        data_train = data[train_begin:train_end]
        normalized_train_data = (
        data_train-np.mean(data_train, axis=0))/np.std(data_train, axis=0)
        train_x, train_y = [], []
        for i in range(len(normalized_train_data)-time_step):
            if i % batch_size == 0:
                batch_index.append(i)
            x = normalized_train_data[i:i+time_step, :j]
            y = normalized_train_data[i:i+time_step, j, np.newaxis]
            train_x.append(x.tolist())
            train_y.append(y.tolist())
        batch_index.append((len(normalized_train_data)-time_step))
        return batch_index, train_x, train_y

    def train_lstm(batch_size=60, time_step=20,epochs=epochs, train_begin=0, train_end=len(data)):
        X = tf.placeholder(tf.float32, shape=[None, time_step, input_size])
        Y = tf.placeholder(tf.float32, shape=[None, time_step, output_size])
        batch_index, train_x, train_y = get_train_data(batch_size, time_step, train_begin, train_end)
        with tf.variable_scope("sec_lstm"):
            pred, _ = lstm(X)
        loss = tf.reduce_mean(
            tf.square(tf.reshape(pred, [-1])-tf.reshape(Y, [-1]))) 
        train_op = tf.train.AdamOptimizer(lr).minimize(loss)  
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=15) 

        with tf.Session() as sess: 
            sess.run(tf.global_variables_initializer())
            for i in range(epochs):
                for step in range(len(batch_index)-1):
                    _, loss_ = sess.run([train_op, loss], feed_dict={X: train_x[batch_index[
                                        step]:batch_index[step+1]], Y: train_y[batch_index[step]:batch_index[step+1]]})
                if (i+1)%50==0:
                    print("Number of epochs:", i+1, " loss:", loss_)
                    print("model_save: ", saver.save(sess, 'model_save/modle.ckpt'))
            print("The train has finished")

# Training begins
    train_lstm()
    
    def get_test_data(time_step=20,data=data,test_begin=0):
        data_test = data[test_begin:]
        mean = np.mean(data_test, axis=0)
        std = np.std(data_test, axis=0)
        normalized_test_data = (data_test-mean)/std
        size = (len(normalized_test_data)+time_step-1)//time_step
        test_x, test_y = [], []
        for i in range(size-1):
            x = normalized_test_data[i*time_step:(i+1)*time_step, :j]
            y = normalized_test_data[i*time_step:(i+1)*time_step, j]
            test_x.append(x.tolist())
            test_y.extend(y)
        test_x.append((normalized_test_data[(i+1)*time_step:, :j]).tolist())
        test_y.extend((normalized_test_data[(i+1)*time_step:, j]).tolist())
        return mean, std, test_x, test_y
 
    def prediction(time_step=1):
        X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
        mean,std,test_x,test_y=get_test_data(time_step,test_begin=0)
        with tf.variable_scope("sec_lstm",reuse=True):
            pred,_=lstm(X)
        saver=tf.train.Saver(tf.global_variables())
        with tf.Session() as sess:
            module_file = tf.train.latest_checkpoint('model_save')
            saver.restore(sess, module_file)
            test_predict=[]
            for step in range(len(test_x)-1):
              prob=sess.run(pred,feed_dict={X:[test_x[step]]})
              predict=prob.reshape((-1))
              test_predict.extend(predict)
            test_y=np.array(test_y)*std[j]+mean[j]
            test_predict=np.array(test_predict)*std[j]+mean[j]
            acc=np.average(np.abs(test_predict-test_y[:len(test_predict)]))
            print("The MAE of this predict:",acc)

# Model training completed, calibration started
    prediction()
    
# Access to data       
    def test_get_test_data(time_step=20,data=test,test_begin=0):
        data_test = data[test_begin:]
        mean = np.mean(data_test, axis=0)
        std = np.std(data_test, axis=0)
        normalized_test_data = (data_test-mean)/std
        size = (len(normalized_test_data)+time_step-1)//time_step
        test_x = []
        for i in range(size-1):
            x = normalized_test_data[i*time_step:(i+1)*time_step, :j]        
            test_x.append(x.tolist())    
        test_x.append((normalized_test_data[(i+1)*time_step:, :j]).tolist())
        return test_x
    def test_prediction(time_step=1):
        X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
        test_x=test_get_test_data(time_step,test_begin=0)
        mean,std,_,_=get_test_data(time_step,test_begin=0)
        with tf.variable_scope("sec_lstm",reuse=True):
            pred,_=lstm(X)
        saver=tf.train.Saver(tf.global_variables())
        with tf.Session() as sess:
            module_file = tf.train.latest_checkpoint('model_save')
            saver.restore(sess, module_file)
            test_predict=[]
            for step in range(len(test_x)-1):
              prob=sess.run(pred,feed_dict={X:[test_x[step]]})
              predict=prob.reshape((-1))
              test_predict.extend(predict)
            test_predict=np.array(test_predict)*std[1]+mean[1]
            return test_predict

    SDtest_predict = test_prediction()
    
# Calculate the correlation coefficient
    tccip_2019 = np.array(tccip_2019)
    tccip_2019 = tccip_2019.flatten()
    SDtest_predict = np.array(SDtest_predict)
    pccs = np.corrcoef(SDtest_predict, tccip_2019)
    locals()['pccs'+str(k)] = pccs [0,1]
    print("data_"+str(k)+"_corr :", locals()['pccs'+str(k)] )

print("Training completed") 

# all_pccs is the correlation coefficient of each combination
all_pccs=[]
for o in range(1,129):
    all_pccs.append(locals()['pccs'+str(o)])
    print()
all_pccs = pd.DataFrame(all_pccs)

# Output(Correlation coefficients of various combinations)
# Please revise the path accordingly
all_pccs.to_csv('/Users/cjchen/Desktop/git-repos/MSWEP_TW/LSTM/output/'+'test_output_8-1_cc.csv',header=False, index=False)