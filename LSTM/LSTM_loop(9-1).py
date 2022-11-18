# Add up to 8 adjacent grids as input at the same time. (9-1:256 combinations)
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
corr_num = np.array([range(1,257)], dtype='float64')
corr_num = corr_num.T
dafT = pd.read_csv(data_path+'test_input_9-1.csv',header=None,names=['year','month','day','1','2','3','4','O','6','7','8','9','TCCIP'])
data = np.array(dafT.iloc[0:1462,2:13])
test = np.array(dafT.iloc[1462:1828:,2:12])
tccip_2019 = np.array(dafT.iloc[1462:1827:,12:13])

# Data extraction
# Traning dataset
#------------------------------------------------------------------------------------------#
# The target grid is used as a separate input for calibration
data1	=	np.array(data[0:1462,[5	,10	]])

# The target grid adds 1 surrounding grid as input for correction at the same time.
data2	=	np.array(data[0:1462,[5,1	,10	]])
data3	=	np.array(data[0:1462,[5,2	,10	]])
data4	=	np.array(data[0:1462,[5,3	,10	]])
data5	=	np.array(data[0:1462,[5,4	,10	]])
data6	=	np.array(data[0:1462,[5,6	,10	]])
data7	=	np.array(data[0:1462,[5,7	,10	]])
data8	=	np.array(data[0:1462,[5,8	,10	]])
data9	=	np.array(data[0:1462,[5,9	,10	]])

# The target grid adds 2 surrounding grid as input for correction at the same time.
data10 =	np.array(data[0:1462,[5,1,2	,10	]])
data11 =	np.array(data[0:1462,[5,1,3	,10	]])
data12 =	np.array(data[0:1462,[5,1,4	,10	]])
data13 =	np.array(data[0:1462,[5,1,6	,10	]])
data14 =	np.array(data[0:1462,[5,1,7	,10	]])
data15 =	np.array(data[0:1462,[5,1,8	,10	]])
data16 =	np.array(data[0:1462,[5,1,9	,10	]])
data17 =	np.array(data[0:1462,[5,2,3	,10	]])
data18 =	np.array(data[0:1462,[5,2,4	,10	]])
data19 =	np.array(data[0:1462,[5,2,6	,10	]])
data20 =	np.array(data[0:1462,[5,2,7	,10	]])
data21 =	np.array(data[0:1462,[5,2,8	,10	]])
data22 =	np.array(data[0:1462,[5,2,9	,10	]])
data23 =	np.array(data[0:1462,[5,3,4	,10	]])
data24 =	np.array(data[0:1462,[5,3,6	,10	]])
data25 =	np.array(data[0:1462,[5,3,7	,10	]])
data26 =	np.array(data[0:1462,[5,3,8	,10	]])
data27 =	np.array(data[0:1462,[5,3,9	,10	]])
data28 =	np.array(data[0:1462,[5,4,6	,10	]])
data29 =	np.array(data[0:1462,[5,4,7	,10	]])
data30 =	np.array(data[0:1462,[5,4,8	,10	]])
data31 =	np.array(data[0:1462,[5,4,9	,10	]])
data32 =	np.array(data[0:1462,[5,6,7	,10	]])
data33 =	np.array(data[0:1462,[5,6,8	,10	]])
data34 =	np.array(data[0:1462,[5,6,9	,10	]])
data35 =	np.array(data[0:1462,[5,7,8	,10	]])
data36 =	np.array(data[0:1462,[5,7,9	,10	]])
data37 =	np.array(data[0:1462,[5,8,9	,10	]])

# The target grid adds 3 surrounding grid as input for correction at the same time.
data38	=	np.array(data[0:1462,[5,1,2,3	,10	]])
data39	=	np.array(data[0:1462,[5,1,2,4	,10	]])
data40	=	np.array(data[0:1462,[5,1,2,6	,10	]])
data41	=	np.array(data[0:1462,[5,1,2,7	,10	]])
data42	=	np.array(data[0:1462,[5,1,2,8	,10	]])
data43	=	np.array(data[0:1462,[5,1,2,9	,10	]])
data44	=	np.array(data[0:1462,[5,1,3,4	,10	]])
data45	=	np.array(data[0:1462,[5,1,3,6	,10	]])
data46	=	np.array(data[0:1462,[5,1,3,7	,10	]])
data47	=	np.array(data[0:1462,[5,1,3,8	,10	]])
data48	=	np.array(data[0:1462,[5,1,3,9	,10	]])
data49	=	np.array(data[0:1462,[5,1,4,6	,10	]])
data50	=	np.array(data[0:1462,[5,1,4,7	,10	]])
data51	=	np.array(data[0:1462,[5,1,4,8	,10	]])
data52	=	np.array(data[0:1462,[5,1,4,9	,10	]])
data53	=	np.array(data[0:1462,[5,1,6,7	,10	]])
data54	=	np.array(data[0:1462,[5,1,6,8	,10	]])
data55	=	np.array(data[0:1462,[5,1,6,9	,10	]])
data56	=	np.array(data[0:1462,[5,1,7,8	,10	]])
data57	=	np.array(data[0:1462,[5,1,7,9	,10	]])
data58	=	np.array(data[0:1462,[5,1,8,9	,10	]])
data59	=	np.array(data[0:1462,[5,2,3,4	,10	]])
data60	=	np.array(data[0:1462,[5,2,3,6	,10	]])
data61	=	np.array(data[0:1462,[5,2,3,7	,10	]])
data62	=	np.array(data[0:1462,[5,2,3,8	,10	]])
data63	=	np.array(data[0:1462,[5,2,3,9	,10	]])
data64	=	np.array(data[0:1462,[5,2,4,6	,10	]])
data65	=	np.array(data[0:1462,[5,2,4,7	,10	]])
data66	=	np.array(data[0:1462,[5,2,4,8	,10	]])
data67	=	np.array(data[0:1462,[5,2,4,9	,10	]])
data68	=	np.array(data[0:1462,[5,2,6,7	,10	]])
data69	=	np.array(data[0:1462,[5,2,6,8	,10	]])
data70	=	np.array(data[0:1462,[5,2,6,9	,10	]])
data71	=	np.array(data[0:1462,[5,2,7,8	,10	]])
data72	=	np.array(data[0:1462,[5,2,7,9	,10	]])
data73	=	np.array(data[0:1462,[5,2,8,9	,10	]])
data74	=	np.array(data[0:1462,[5,3,4,6	,10	]])
data75	=	np.array(data[0:1462,[5,3,4,7	,10	]])
data76	=	np.array(data[0:1462,[5,3,4,8	,10	]])
data77	=	np.array(data[0:1462,[5,3,4,9	,10	]])
data78	=	np.array(data[0:1462,[5,3,6,7	,10	]])
data79	=	np.array(data[0:1462,[5,3,6,8	,10	]])
data80	=	np.array(data[0:1462,[5,3,6,9	,10	]])
data81	=	np.array(data[0:1462,[5,3,7,8	,10	]])
data82	=	np.array(data[0:1462,[5,3,7,9	,10	]])
data83	=	np.array(data[0:1462,[5,3,8,9	,10	]])
data84	=	np.array(data[0:1462,[5,4,6,7	,10	]])
data85	=	np.array(data[0:1462,[5,4,6,8	,10	]])
data86	=	np.array(data[0:1462,[5,4,6,9	,10	]])
data87	=	np.array(data[0:1462,[5,4,7,8	,10	]])
data88	=	np.array(data[0:1462,[5,4,7,9	,10	]])
data89	=	np.array(data[0:1462,[5,4,8,9	,10	]])
data90	=	np.array(data[0:1462,[5,6,7,8	,10	]])
data91	=	np.array(data[0:1462,[5,6,7,9	,10	]])
data92	=	np.array(data[0:1462,[5,6,8,9	,10	]])
data93	=	np.array(data[0:1462,[5,7,8,9	,10	]])

# The target grid adds 4 surrounding grid as input for correction at the same time.
data94 	=	np.array(data[0:1462,[5,1,2,3,4	,10	]])
data95 	=	np.array(data[0:1462,[5,1,2,3,6	,10	]])
data96 	=	np.array(data[0:1462,[5,1,2,3,7	,10	]])
data97 	=	np.array(data[0:1462,[5,1,2,3,8	,10	]])
data98 	=	np.array(data[0:1462,[5,1,2,3,8	,10	]])
data99 	=	np.array(data[0:1462,[5,1,2,4,6	,10	]])
data100	=	np.array(data[0:1462,[5,1,2,4,7	,10	]])
data101	=	np.array(data[0:1462,[5,1,2,4,8	,10	]])
data102	=	np.array(data[0:1462,[5,1,2,4,9	,10	]])
data103	=	np.array(data[0:1462,[5,1,2,6,7	,10	]])
data104	=	np.array(data[0:1462,[5,1,2,6,8	,10	]])
data105	=	np.array(data[0:1462,[5,1,2,6,9	,10	]])
data106	=	np.array(data[0:1462,[5,1,2,7,8	,10	]])
data107	=	np.array(data[0:1462,[5,1,2,7,9	,10	]])
data108	=	np.array(data[0:1462,[5,1,2,8,9	,10	]])
data109	=	np.array(data[0:1462,[5,1,3,4,6	,10	]])
data110	=	np.array(data[0:1462,[5,1,3,4,7	,10	]])
data111	=	np.array(data[0:1462,[5,1,3,4,8	,10	]])
data112	=	np.array(data[0:1462,[5,1,3,4,9	,10	]])
data113	=	np.array(data[0:1462,[5,1,3,6,7	,10	]])
data114	=	np.array(data[0:1462,[5,1,3,6,8	,10	]])
data115	=	np.array(data[0:1462,[5,1,3,6,9	,10	]])
data116	=	np.array(data[0:1462,[5,1,3,7,8	,10	]])
data117	=	np.array(data[0:1462,[5,1,3,7,9	,10	]])
data118	=	np.array(data[0:1462,[5,1,3,8,9	,10	]])
data119	=	np.array(data[0:1462,[5,1,4,6,7	,10	]])
data120	=	np.array(data[0:1462,[5,1,4,6,8	,10	]])
data121	=	np.array(data[0:1462,[5,1,4,6,9	,10	]])
data122	=	np.array(data[0:1462,[5,1,4,7,8	,10	]])
data123	=	np.array(data[0:1462,[5,1,4,7,9	,10	]])
data124	=	np.array(data[0:1462,[5,1,4,8,9	,10	]])
data125	=	np.array(data[0:1462,[5,1,6,7,8	,10	]])
data126	=	np.array(data[0:1462,[5,1,6,7,9	,10	]])
data127	=	np.array(data[0:1462,[5,1,6,8,9	,10	]])
data128	=	np.array(data[0:1462,[5,1,7,8,9	,10	]])
data129	=	np.array(data[0:1462,[5,2,3,4,6	,10	]])
data130	=	np.array(data[0:1462,[5,2,3,4,7	,10	]])
data131	=	np.array(data[0:1462,[5,2,3,4,8	,10	]])
data132	=	np.array(data[0:1462,[5,2,3,4,9	,10	]])
data133	=	np.array(data[0:1462,[5,2,3,6,7	,10	]])
data134	=	np.array(data[0:1462,[5,2,3,6,8	,10	]])
data135	=	np.array(data[0:1462,[5,2,3,6,9	,10	]])
data136	=	np.array(data[0:1462,[5,2,3,7,8	,10	]])
data137	=	np.array(data[0:1462,[5,2,3,7,9	,10	]])
data138	=	np.array(data[0:1462,[5,2,3,8,9	,10	]])
data139	=	np.array(data[0:1462,[5,2,4,6,7	,10	]])
data140	=	np.array(data[0:1462,[5,2,4,6,8	,10	]])
data141	=	np.array(data[0:1462,[5,2,4,6,9	,10	]])
data142	=	np.array(data[0:1462,[5,2,4,7,8	,10	]])
data143	=	np.array(data[0:1462,[5,2,4,7,9	,10	]])
data144	=	np.array(data[0:1462,[5,2,4,8,9	,10	]])
data145	=	np.array(data[0:1462,[5,2,6,7,8	,10	]])
data146	=	np.array(data[0:1462,[5,2,6,7,9	,10	]])
data147	=	np.array(data[0:1462,[5,2,6,8,9	,10	]])
data148	=	np.array(data[0:1462,[5,2,7,8,9	,10	]])
data149	=	np.array(data[0:1462,[5,3,4,6,7	,10	]])
data150	=	np.array(data[0:1462,[5,3,4,6,8	,10	]])
data151	=	np.array(data[0:1462,[5,3,4,6,9	,10	]])
data152	=	np.array(data[0:1462,[5,3,4,7,8	,10	]])
data153	=	np.array(data[0:1462,[5,3,4,7,9	,10	]])
data154	=	np.array(data[0:1462,[5,3,4,8,9	,10	]])
data155	=	np.array(data[0:1462,[5,3,6,7,8	,10	]])
data156	=	np.array(data[0:1462,[5,3,6,7,9	,10	]])
data157	=	np.array(data[0:1462,[5,3,6,8,9	,10	]])
data158	=	np.array(data[0:1462,[5,3,7,8,9	,10	]])
data159	=	np.array(data[0:1462,[5,4,6,7,8	,10	]])
data160	=	np.array(data[0:1462,[5,4,6,7,9	,10	]])
data161	=	np.array(data[0:1462,[5,4,6,8,9	,10	]])
data162	=	np.array(data[0:1462,[5,4,7,8,9	,10	]])
data163	=	np.array(data[0:1462,[5,6,7,8,9	,10	]])

# The target grid adds 5 surrounding grid as input for correction at the same time.
data164	=	np.array(data[0:1462,[5,1,2,3,4,6	,10	]])
data165	=	np.array(data[0:1462,[5,1,2,3,4,7	,10	]])
data166	=	np.array(data[0:1462,[5,1,2,3,4,8	,10	]])
data167	=	np.array(data[0:1462,[5,1,2,3,4,9	,10	]])
data168	=	np.array(data[0:1462,[5,1,2,3,6,7	,10	]])
data169	=	np.array(data[0:1462,[5,1,2,3,6,8	,10	]])
data170	=	np.array(data[0:1462,[5,1,2,3,6,9	,10	]])
data171	=	np.array(data[0:1462,[5,1,2,3,7,8	,10	]])
data172	=	np.array(data[0:1462,[5,1,2,3,7,9	,10	]])
data173	=	np.array(data[0:1462,[5,1,2,3,8,9	,10	]])
data174	=	np.array(data[0:1462,[5,1,2,4,6,7	,10	]])
data175	=	np.array(data[0:1462,[5,1,2,4,6,8	,10	]])
data176	=	np.array(data[0:1462,[5,1,2,4,6,9	,10	]])
data177	=	np.array(data[0:1462,[5,1,2,4,7,8	,10	]])
data178	=	np.array(data[0:1462,[5,1,2,4,7,9	,10	]])
data179	=	np.array(data[0:1462,[5,1,2,4,8,9	,10	]])
data180	=	np.array(data[0:1462,[5,1,2,6,7,8	,10	]])
data181	=	np.array(data[0:1462,[5,1,2,6,7,9	,10	]])
data182	=	np.array(data[0:1462,[5,1,2,6,8,9	,10	]])
data183	=	np.array(data[0:1462,[5,1,2,7,8,9	,10	]])
data184	=	np.array(data[0:1462,[5,1,3,4,6,7	,10	]])
data185	=	np.array(data[0:1462,[5,1,3,4,6,8	,10	]])
data186	=	np.array(data[0:1462,[5,1,3,4,6,9	,10	]])
data187	=	np.array(data[0:1462,[5,1,3,4,7,8	,10	]])
data188	=	np.array(data[0:1462,[5,1,3,4,7,9	,10	]])
data189	=	np.array(data[0:1462,[5,1,3,4,8,9	,10	]])
data190	=	np.array(data[0:1462,[5,1,3,6,7,8	,10	]])
data191	=	np.array(data[0:1462,[5,1,3,6,7,9	,10	]])
data192	=	np.array(data[0:1462,[5,1,3,6,8,9	,10	]])
data193	=	np.array(data[0:1462,[5,1,3,7,8,9	,10	]])
data194	=	np.array(data[0:1462,[5,1,4,6,7,8	,10	]])
data195	=	np.array(data[0:1462,[5,1,4,6,7,9	,10	]])
data196	=	np.array(data[0:1462,[5,1,4,6,8,9	,10	]])
data197	=	np.array(data[0:1462,[5,1,4,7,8,9	,10	]])
data198	=	np.array(data[0:1462,[5,1,6,7,8,9	,10	]])
data199	=	np.array(data[0:1462,[5,2,3,4,6,7	,10	]])
data200	=	np.array(data[0:1462,[5,2,3,4,6,8	,10	]])
data201	=	np.array(data[0:1462,[5,2,3,4,6,9	,10	]])
data202	=	np.array(data[0:1462,[5,2,3,4,7,8	,10	]])
data203	=	np.array(data[0:1462,[5,2,3,4,7,9	,10	]])
data204	=	np.array(data[0:1462,[5,2,3,4,8,9	,10	]])
data205	=	np.array(data[0:1462,[5,2,3,6,7,8	,10	]])
data206	=	np.array(data[0:1462,[5,2,3,6,7,9	,10	]])
data207	=	np.array(data[0:1462,[5,2,3,6,8,9	,10	]])
data208	=	np.array(data[0:1462,[5,2,3,7,8,9	,10	]])
data209	=	np.array(data[0:1462,[5,2,4,6,7,8	,10	]])
data210	=	np.array(data[0:1462,[5,2,4,6,7,9	,10	]])
data211	=	np.array(data[0:1462,[5,2,4,6,8,9	,10	]])
data212	=	np.array(data[0:1462,[5,2,4,7,8,9	,10	]])
data213	=	np.array(data[0:1462,[5,2,6,7,8,9	,10	]])
data214	=	np.array(data[0:1462,[5,3,4,6,7,8	,10	]])
data215	=	np.array(data[0:1462,[5,3,4,6,7,9	,10	]])
data216	=	np.array(data[0:1462,[5,3,4,6,8,9	,10	]])
data217	=	np.array(data[0:1462,[5,3,4,7,8,9	,10	]])
data218	=	np.array(data[0:1462,[5,3,6,7,8,9	,10	]])
data219	=	np.array(data[0:1462,[5,4,6,7,8,9	,10	]])

# The target grid adds 6 surrounding grid as input for correction at the same time.
data220	=	np.array(data[0:1462,[5,1,2,3,4,6,7	,10	]])
data221	=	np.array(data[0:1462,[5,1,2,3,4,6,8	,10	]])
data222	=	np.array(data[0:1462,[5,1,2,3,4,6,9	,10	]])
data223	=	np.array(data[0:1462,[5,1,2,3,4,7,8	,10	]])
data224	=	np.array(data[0:1462,[5,1,2,3,4,7,9	,10	]])
data225	=	np.array(data[0:1462,[5,1,2,3,4,8,9	,10	]])
data226	=	np.array(data[0:1462,[5,1,2,3,6,7,8	,10	]])
data227	=	np.array(data[0:1462,[5,1,2,3,6,7,9	,10	]])
data228	=	np.array(data[0:1462,[5,1,2,3,6,8,9	,10	]])
data229	=	np.array(data[0:1462,[5,1,2,3,7,8,9	,10	]])
data230	=	np.array(data[0:1462,[5,1,2,4,6,7,8	,10	]])
data231	=	np.array(data[0:1462,[5,1,2,4,6,7,9	,10	]])
data232	=	np.array(data[0:1462,[5,1,2,4,6,8,9	,10	]])
data233	=	np.array(data[0:1462,[5,1,2,4,7,8,9	,10	]])
data234	=	np.array(data[0:1462,[5,1,2,4,7,8,9	,10	]])
data235	=	np.array(data[0:1462,[5,1,3,4,6,7,8	,10	]])
data236	=	np.array(data[0:1462,[5,1,3,4,6,7,9	,10	]])
data237	=	np.array(data[0:1462,[5,1,3,4,6,8,9	,10	]])
data238	=	np.array(data[0:1462,[5,1,3,4,7,8,9	,10	]])
data239	=	np.array(data[0:1462,[5,1,3,6,7,8,9	,10	]])
data240	=	np.array(data[0:1462,[5,1,4,6,7,8,9	,10	]])
data241	=	np.array(data[0:1462,[5,2,3,4,6,7,8	,10	]])
data242	=	np.array(data[0:1462,[5,2,3,4,6,7,9	,10	]])
data243	=	np.array(data[0:1462,[5,2,3,4,6,8,9	,10	]])
data244	=	np.array(data[0:1462,[5,2,3,4,7,8,9	,10	]])
data245	=	np.array(data[0:1462,[5,2,3,6,7,8,9	,10	]])
data246	=	np.array(data[0:1462,[5,2,4,6,7,8,9	,10	]])
data247	=	np.array(data[0:1462,[5,3,4,6,7,8,9	,10	]])

# The target grid adds 7 surrounding grid as input for correction at the same time.
data248	=	np.array(data[0:1462,[5,1,2,3,4,6,7,8	,10	]])
data249	=	np.array(data[0:1462,[5,1,2,3,4,6,7,9	,10	]])
data250	=	np.array(data[0:1462,[5,1,2,3,4,6,8,9	,10	]])
data251	=	np.array(data[0:1462,[5,1,2,3,4,7,8,9	,10	]])
data252	=	np.array(data[0:1462,[5,1,2,3,6,7,8,9	,10	]])
data253	=	np.array(data[0:1462,[5,1,2,4,6,7,8,9	,10	]])
data254	=	np.array(data[0:1462,[5,1,3,4,6,7,8,9	,10	]])
data255	=	np.array(data[0:1462,[5,2,3,4,6,7,8,9	,10	]])

# The target grid adds 8 surrounding grid as input for correction at the same time.
data256	=	np.array(data[0:1462,[5,1,2,3,4,6,7,8,9	,10	]])
#------------------------------------------------------------------------------------------#

# Validation dataset
#------------------------------------------------------------------------------------------#
test1	=	np.array(test[0:366,[5		]])

test2	=	np.array(test[0:366,[5,1		]])
test3	=	np.array(test[0:366,[5,2		]])
test4	=	np.array(test[0:366,[5,3		]])
test5	=	np.array(test[0:366,[5,4		]])
test6	=	np.array(test[0:366,[5,6		]])
test7	=	np.array(test[0:366,[5,7		]])
test8	=	np.array(test[0:366,[5,8		]])
test9	=	np.array(test[0:366,[5,9		]])

test10 =	np.array(test[0:366,[5,1,2		]])
test11 =	np.array(test[0:366,[5,1,3		]])
test12 =	np.array(test[0:366,[5,1,4		]])
test13 =	np.array(test[0:366,[5,1,6		]])
test14 =	np.array(test[0:366,[5,1,7		]])
test15 =	np.array(test[0:366,[5,1,8		]])
test16 =	np.array(test[0:366,[5,1,9		]])
test17 =	np.array(test[0:366,[5,2,3		]])
test18 =	np.array(test[0:366,[5,2,4		]])
test19 =	np.array(test[0:366,[5,2,6		]])
test20 =	np.array(test[0:366,[5,2,7		]])
test21 =	np.array(test[0:366,[5,2,8		]])
test22 =	np.array(test[0:366,[5,2,9		]])
test23 =	np.array(test[0:366,[5,3,4		]])
test24 =	np.array(test[0:366,[5,3,6		]])
test25 =	np.array(test[0:366,[5,3,7		]])
test26 =	np.array(test[0:366,[5,3,8		]])
test27 =	np.array(test[0:366,[5,3,9		]])
test28 =	np.array(test[0:366,[5,4,6		]])
test29 =	np.array(test[0:366,[5,4,7		]])
test30 =	np.array(test[0:366,[5,4,8		]])
test31 =	np.array(test[0:366,[5,4,9		]])
test32 =	np.array(test[0:366,[5,6,7		]])
test33 =	np.array(test[0:366,[5,6,8		]])
test34 =	np.array(test[0:366,[5,6,9		]])
test35 =	np.array(test[0:366,[5,7,8		]])
test36 =	np.array(test[0:366,[5,7,9		]])
test37 =	np.array(test[0:366,[5,8,9		]])

test38	=	np.array(test[0:366,[5,1,2,3		]])
test39	=	np.array(test[0:366,[5,1,2,4		]])
test40	=	np.array(test[0:366,[5,1,2,6		]])
test41	=	np.array(test[0:366,[5,1,2,7		]])
test42	=	np.array(test[0:366,[5,1,2,8		]])
test43	=	np.array(test[0:366,[5,1,2,9		]])
test44	=	np.array(test[0:366,[5,1,3,4		]])
test45	=	np.array(test[0:366,[5,1,3,6		]])
test46	=	np.array(test[0:366,[5,1,3,7		]])
test47	=	np.array(test[0:366,[5,1,3,8		]])
test48	=	np.array(test[0:366,[5,1,3,9		]])
test49	=	np.array(test[0:366,[5,1,4,6		]])
test50	=	np.array(test[0:366,[5,1,4,7		]])
test51	=	np.array(test[0:366,[5,1,4,8		]])
test52	=	np.array(test[0:366,[5,1,4,9		]])
test53	=	np.array(test[0:366,[5,1,6,7		]])
test54	=	np.array(test[0:366,[5,1,6,8		]])
test55	=	np.array(test[0:366,[5,1,6,9		]])
test56	=	np.array(test[0:366,[5,1,7,8		]])
test57	=	np.array(test[0:366,[5,1,7,9		]])
test58	=	np.array(test[0:366,[5,1,8,9		]])
test59	=	np.array(test[0:366,[5,2,3,4		]])
test60	=	np.array(test[0:366,[5,2,3,6		]])
test61	=	np.array(test[0:366,[5,2,3,7		]])
test62	=	np.array(test[0:366,[5,2,3,8		]])
test63	=	np.array(test[0:366,[5,2,3,9		]])
test64	=	np.array(test[0:366,[5,2,4,6		]])
test65	=	np.array(test[0:366,[5,2,4,7		]])
test66	=	np.array(test[0:366,[5,2,4,8		]])
test67	=	np.array(test[0:366,[5,2,4,9		]])
test68	=	np.array(test[0:366,[5,2,6,7		]])
test69	=	np.array(test[0:366,[5,2,6,8		]])
test70	=	np.array(test[0:366,[5,2,6,9		]])
test71	=	np.array(test[0:366,[5,2,7,8		]])
test72	=	np.array(test[0:366,[5,2,7,9		]])
test73	=	np.array(test[0:366,[5,2,8,9		]])
test74	=	np.array(test[0:366,[5,3,4,6		]])
test75	=	np.array(test[0:366,[5,3,4,7		]])
test76	=	np.array(test[0:366,[5,3,4,8		]])
test77	=	np.array(test[0:366,[5,3,4,9		]])
test78	=	np.array(test[0:366,[5,3,6,7		]])
test79	=	np.array(test[0:366,[5,3,6,8		]])
test80	=	np.array(test[0:366,[5,3,6,9		]])
test81	=	np.array(test[0:366,[5,3,7,8		]])
test82	=	np.array(test[0:366,[5,3,7,9		]])
test83	=	np.array(test[0:366,[5,3,8,9		]])
test84	=	np.array(test[0:366,[5,4,6,7		]])
test85	=	np.array(test[0:366,[5,4,6,8		]])
test86	=	np.array(test[0:366,[5,4,6,9		]])
test87	=	np.array(test[0:366,[5,4,7,8		]])
test88	=	np.array(test[0:366,[5,4,7,9		]])
test89	=	np.array(test[0:366,[5,4,8,9		]])
test90	=	np.array(test[0:366,[5,6,7,8		]])
test91	=	np.array(test[0:366,[5,6,7,9		]])
test92	=	np.array(test[0:366,[5,6,8,9		]])
test93	=	np.array(test[0:366,[5,7,8,9		]])

test94 	=	np.array(test[0:366,[5,1,2,3,4		]])
test95 	=	np.array(test[0:366,[5,1,2,3,6		]])
test96 	=	np.array(test[0:366,[5,1,2,3,7		]])
test97 	=	np.array(test[0:366,[5,1,2,3,8		]])
test98 	=	np.array(test[0:366,[5,1,2,3,8		]])
test99 	=	np.array(test[0:366,[5,1,2,4,6		]])
test100	=	np.array(test[0:366,[5,1,2,4,7		]])
test101	=	np.array(test[0:366,[5,1,2,4,8		]])
test102	=	np.array(test[0:366,[5,1,2,4,9		]])
test103	=	np.array(test[0:366,[5,1,2,6,7		]])
test104	=	np.array(test[0:366,[5,1,2,6,8		]])
test105	=	np.array(test[0:366,[5,1,2,6,9		]])
test106	=	np.array(test[0:366,[5,1,2,7,8		]])
test107	=	np.array(test[0:366,[5,1,2,7,9		]])
test108	=	np.array(test[0:366,[5,1,2,8,9		]])
test109	=	np.array(test[0:366,[5,1,3,4,6		]])
test110	=	np.array(test[0:366,[5,1,3,4,7		]])
test111	=	np.array(test[0:366,[5,1,3,4,8		]])
test112	=	np.array(test[0:366,[5,1,3,4,9		]])
test113	=	np.array(test[0:366,[5,1,3,6,7		]])
test114	=	np.array(test[0:366,[5,1,3,6,8		]])
test115	=	np.array(test[0:366,[5,1,3,6,9		]])
test116	=	np.array(test[0:366,[5,1,3,7,8		]])
test117	=	np.array(test[0:366,[5,1,3,7,9		]])
test118	=	np.array(test[0:366,[5,1,3,8,9		]])
test119	=	np.array(test[0:366,[5,1,4,6,7		]])
test120	=	np.array(test[0:366,[5,1,4,6,8		]])
test121	=	np.array(test[0:366,[5,1,4,6,9		]])
test122	=	np.array(test[0:366,[5,1,4,7,8		]])
test123	=	np.array(test[0:366,[5,1,4,7,9		]])
test124	=	np.array(test[0:366,[5,1,4,8,9		]])
test125	=	np.array(test[0:366,[5,1,6,7,8		]])
test126	=	np.array(test[0:366,[5,1,6,7,9		]])
test127	=	np.array(test[0:366,[5,1,6,8,9		]])
test128	=	np.array(test[0:366,[5,1,7,8,9		]])
test129	=	np.array(test[0:366,[5,2,3,4,6		]])
test130	=	np.array(test[0:366,[5,2,3,4,7		]])
test131	=	np.array(test[0:366,[5,2,3,4,8		]])
test132	=	np.array(test[0:366,[5,2,3,4,9		]])
test133	=	np.array(test[0:366,[5,2,3,6,7		]])
test134	=	np.array(test[0:366,[5,2,3,6,8		]])
test135	=	np.array(test[0:366,[5,2,3,6,9		]])
test136	=	np.array(test[0:366,[5,2,3,7,8		]])
test137	=	np.array(test[0:366,[5,2,3,7,9		]])
test138	=	np.array(test[0:366,[5,2,3,8,9		]])
test139	=	np.array(test[0:366,[5,2,4,6,7		]])
test140	=	np.array(test[0:366,[5,2,4,6,8		]])
test141	=	np.array(test[0:366,[5,2,4,6,9		]])
test142	=	np.array(test[0:366,[5,2,4,7,8		]])
test143	=	np.array(test[0:366,[5,2,4,7,9		]])
test144	=	np.array(test[0:366,[5,2,4,8,9		]])
test145	=	np.array(test[0:366,[5,2,6,7,8		]])
test146	=	np.array(test[0:366,[5,2,6,7,9		]])
test147	=	np.array(test[0:366,[5,2,6,8,9		]])
test148	=	np.array(test[0:366,[5,2,7,8,9		]])
test149	=	np.array(test[0:366,[5,3,4,6,7		]])
test150	=	np.array(test[0:366,[5,3,4,6,8		]])
test151	=	np.array(test[0:366,[5,3,4,6,9		]])
test152	=	np.array(test[0:366,[5,3,4,7,8		]])
test153	=	np.array(test[0:366,[5,3,4,7,9		]])
test154	=	np.array(test[0:366,[5,3,4,8,9		]])
test155	=	np.array(test[0:366,[5,3,6,7,8		]])
test156	=	np.array(test[0:366,[5,3,6,7,9		]])
test157	=	np.array(test[0:366,[5,3,6,8,9		]])
test158	=	np.array(test[0:366,[5,3,7,8,9		]])
test159	=	np.array(test[0:366,[5,4,6,7,8		]])
test160	=	np.array(test[0:366,[5,4,6,7,9		]])
test161	=	np.array(test[0:366,[5,4,6,8,9		]])
test162	=	np.array(test[0:366,[5,4,7,8,9		]])
test163	=	np.array(test[0:366,[5,6,7,8,9		]])

test164	=	np.array(test[0:366,[5,1,2,3,4,6		]])
test165	=	np.array(test[0:366,[5,1,2,3,4,7		]])
test166	=	np.array(test[0:366,[5,1,2,3,4,8		]])
test167	=	np.array(test[0:366,[5,1,2,3,4,9		]])
test168	=	np.array(test[0:366,[5,1,2,3,6,7		]])
test169	=	np.array(test[0:366,[5,1,2,3,6,8		]])
test170	=	np.array(test[0:366,[5,1,2,3,6,9		]])
test171	=	np.array(test[0:366,[5,1,2,3,7,8		]])
test172	=	np.array(test[0:366,[5,1,2,3,7,9		]])
test173	=	np.array(test[0:366,[5,1,2,3,8,9		]])
test174	=	np.array(test[0:366,[5,1,2,4,6,7		]])
test175	=	np.array(test[0:366,[5,1,2,4,6,8		]])
test176	=	np.array(test[0:366,[5,1,2,4,6,9		]])
test177	=	np.array(test[0:366,[5,1,2,4,7,8		]])
test178	=	np.array(test[0:366,[5,1,2,4,7,9		]])
test179	=	np.array(test[0:366,[5,1,2,4,8,9		]])
test180	=	np.array(test[0:366,[5,1,2,6,7,8		]])
test181	=	np.array(test[0:366,[5,1,2,6,7,9		]])
test182	=	np.array(test[0:366,[5,1,2,6,8,9		]])
test183	=	np.array(test[0:366,[5,1,2,7,8,9		]])
test184	=	np.array(test[0:366,[5,1,3,4,6,7		]])
test185	=	np.array(test[0:366,[5,1,3,4,6,8		]])
test186	=	np.array(test[0:366,[5,1,3,4,6,9		]])
test187	=	np.array(test[0:366,[5,1,3,4,7,8		]])
test188	=	np.array(test[0:366,[5,1,3,4,7,9		]])
test189	=	np.array(test[0:366,[5,1,3,4,8,9		]])
test190	=	np.array(test[0:366,[5,1,3,6,7,8		]])
test191	=	np.array(test[0:366,[5,1,3,6,7,9		]])
test192	=	np.array(test[0:366,[5,1,3,6,8,9		]])
test193	=	np.array(test[0:366,[5,1,3,7,8,9		]])
test194	=	np.array(test[0:366,[5,1,4,6,7,8		]])
test195	=	np.array(test[0:366,[5,1,4,6,7,9		]])
test196	=	np.array(test[0:366,[5,1,4,6,8,9		]])
test197	=	np.array(test[0:366,[5,1,4,7,8,9		]])
test198	=	np.array(test[0:366,[5,1,6,7,8,9		]])
test199	=	np.array(test[0:366,[5,2,3,4,6,7		]])
test200	=	np.array(test[0:366,[5,2,3,4,6,8		]])
test201	=	np.array(test[0:366,[5,2,3,4,6,9		]])
test202	=	np.array(test[0:366,[5,2,3,4,7,8		]])
test203	=	np.array(test[0:366,[5,2,3,4,7,9		]])
test204	=	np.array(test[0:366,[5,2,3,4,8,9		]])
test205	=	np.array(test[0:366,[5,2,3,6,7,8		]])
test206	=	np.array(test[0:366,[5,2,3,6,7,9		]])
test207	=	np.array(test[0:366,[5,2,3,6,8,9		]])
test208	=	np.array(test[0:366,[5,2,3,7,8,9		]])
test209	=	np.array(test[0:366,[5,2,4,6,7,8		]])
test210	=	np.array(test[0:366,[5,2,4,6,7,9		]])
test211	=	np.array(test[0:366,[5,2,4,6,8,9		]])
test212	=	np.array(test[0:366,[5,2,4,7,8,9		]])
test213	=	np.array(test[0:366,[5,2,6,7,8,9		]])
test214	=	np.array(test[0:366,[5,3,4,6,7,8		]])
test215	=	np.array(test[0:366,[5,3,4,6,7,9		]])
test216	=	np.array(test[0:366,[5,3,4,6,8,9		]])
test217	=	np.array(test[0:366,[5,3,4,7,8,9		]])
test218	=	np.array(test[0:366,[5,3,6,7,8,9		]])
test219	=	np.array(test[0:366,[5,4,6,7,8,9		]])

test220	=	np.array(test[0:366,[5,1,2,3,4,6,7		]])
test221	=	np.array(test[0:366,[5,1,2,3,4,6,8		]])
test222	=	np.array(test[0:366,[5,1,2,3,4,6,9		]])
test223	=	np.array(test[0:366,[5,1,2,3,4,7,8		]])
test224	=	np.array(test[0:366,[5,1,2,3,4,7,9		]])
test225	=	np.array(test[0:366,[5,1,2,3,4,8,9		]])
test226	=	np.array(test[0:366,[5,1,2,3,6,7,8		]])
test227	=	np.array(test[0:366,[5,1,2,3,6,7,9		]])
test228	=	np.array(test[0:366,[5,1,2,3,6,8,9		]])
test229	=	np.array(test[0:366,[5,1,2,3,7,8,9		]])
test230	=	np.array(test[0:366,[5,1,2,4,6,7,8		]])
test231	=	np.array(test[0:366,[5,1,2,4,6,7,9		]])
test232	=	np.array(test[0:366,[5,1,2,4,6,8,9		]])
test233	=	np.array(test[0:366,[5,1,2,4,7,8,9		]])
test234	=	np.array(test[0:366,[5,1,2,4,7,8,9		]])
test235	=	np.array(test[0:366,[5,1,3,4,6,7,8		]])
test236	=	np.array(test[0:366,[5,1,3,4,6,7,9		]])
test237	=	np.array(test[0:366,[5,1,3,4,6,8,9		]])
test238	=	np.array(test[0:366,[5,1,3,4,7,8,9		]])
test239	=	np.array(test[0:366,[5,1,3,6,7,8,9		]])
test240	=	np.array(test[0:366,[5,1,4,6,7,8,9		]])
test241	=	np.array(test[0:366,[5,2,3,4,6,7,8		]])
test242	=	np.array(test[0:366,[5,2,3,4,6,7,9		]])
test243	=	np.array(test[0:366,[5,2,3,4,6,8,9		]])
test244	=	np.array(test[0:366,[5,2,3,4,7,8,9		]])
test245	=	np.array(test[0:366,[5,2,3,6,7,8,9		]])
test246	=	np.array(test[0:366,[5,2,4,6,7,8,9		]])
test247	=	np.array(test[0:366,[5,3,4,6,7,8,9		]])

test248	=	np.array(test[0:366,[5,1,2,3,4,6,7,8		]])
test249	=	np.array(test[0:366,[5,1,2,3,4,6,7,9		]])
test250	=	np.array(test[0:366,[5,1,2,3,4,6,8,9		]])
test251	=	np.array(test[0:366,[5,1,2,3,4,7,8,9		]])
test252	=	np.array(test[0:366,[5,1,2,3,6,7,8,9		]])
test253	=	np.array(test[0:366,[5,1,2,4,6,7,8,9		]])
test254	=	np.array(test[0:366,[5,1,3,4,6,7,8,9		]])
test255	=	np.array(test[0:366,[5,2,3,4,6,7,8,9		]])

test256	=	np.array(test[0:366,[5,1,2,3,4,6,7,8,9		]])
#------------------------------------------------------------------------------------------#

# Variable k is the file number 1~256.
# The variable j is entered and other parameters. If for file k=1,j=1. k=2~9,j=2. etc.
#    1    = 1 ( 1  )
# 2  ~  9 = 2 ( 8  )
# 10 ~ 37 = 3 ( 28 )
# 38 ~ 93 = 4 ( 56 )
# 94 ~163 = 5 ( 70 )
# 164~219 = 6 ( 56 )
# 220~247 = 7 ( 28 )
# 248~255 = 8 ( 8  )
#   256   = 9 ( 1  )
for k in range(1,257):
    if k == 1 :
         j = 1;
    elif 2   <= k <=   9:
         j = 2;
    elif 10  <= k <=  37:
         j = 3;
    elif 38  <= k <=  93:
         j = 4;
    elif 94  <= k <= 163:
         j = 5;
    elif 164 <= k <= 219:
         j = 6;
    elif 220 <= k <= 247:
         j = 7;
    elif 248 <= k <= 255:
         j = 8;
    elif k == 256:
         j = 9;
    
    
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
for o in range(1,257):
    all_pccs.append(locals()['pccs'+str(o)])
    print()
all_pccs = pd.DataFrame(all_pccs)

# Output(Correlation coefficients of various combinations)
# Please revise the path accordingly
all_pccs.to_csv('/Users/cjchen/Desktop/git-repos/MSWEP_TW/LSTM/output/'+'test_output_9-1_cc.csv',header=False, index=False)