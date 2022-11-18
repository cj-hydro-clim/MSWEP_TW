# Add up to 1 adjacent grids as input at the same time. (2-1:2 combinations)
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
corr_num = np.array([range(1,3)], dtype='float64')
corr_num = corr_num.T
dafT = pd.read_csv(data_path+'test_input_2-1.csv',header=None,names=['year','month','day','1','2','TCCIP'])
data = np.array(dafT.iloc[0:1462,2:6])
test = np.array(dafT.iloc[1462:1828:,2:5])
tccip_2019 = np.array(dafT.iloc[1462:1827:,5:6])

# Data extraction
# Traning dataset
#------------------------------------------------------------------------------------------#
# The target grid is used as a separate input for calibration
data1	=	np.array(data[0:1462,[1	,3	]])

# The target grid adds 1 surrounding grid as input for correction at the same time.
data2	=	np.array(data[0:1462,[1,2	,3 	]])
#------------------------------------------------------------------------------------------#

# Validation dataset
#------------------------------------------------------------------------------------------#
test1	=	np.array(test[0:366,[1		]])

test2	=	np.array(test[0:366,[1,2	 	]])
#------------------------------------------------------------------------------------------#

# Variable k is the file number 1~2.
# The variable j is entered and other parameters. If for file k=1,j=1. k=2,j=2.
#    1    = 1 ( 1 )
#    2    = 2 ( 1 )
for k in range(1,3):
    if k == 1 :
         j = 1;
    elif k == 2:
         j = 2;


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
for o in range(1,3):
    all_pccs.append(locals()['pccs'+str(o)])
    print()
all_pccs = pd.DataFrame(all_pccs)

# Output(Correlation coefficients of various combinations)
# Please revise the path accordingly
all_pccs.to_csv('/Users/cjchen/Desktop/git-repos/MSWEP_TW/LSTM/output/'+'test_output_2-1_cc.csv',header=False, index=False)
