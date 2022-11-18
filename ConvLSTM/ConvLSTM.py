import random
import numpy
import pandas as pd
from pandas import read_csv
import math
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import ConvLSTM2D
from tensorflow.keras.layers import Flatten
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
tf.reset_default_graph()

# Random seed
random.seed(718)
numpy.random.seed(718)
tf.set_random_seed(718)

# Please revise the data_path accordingly
data_path = '/Users/cjchen/Desktop/git-repos/MSWEP_TW/ConvLSTM/input/'
# Precipitation data to be corrected (IMERG)
dataframe = read_csv(data_path+'test_in_1.csv',header=None, engine='python')
# Precipitation data for correction basis (TCCIP)
datav = read_csv(data_path+'test_in_2.csv',header=None, engine='python')

dataset = dataframe.values
datav = datav.values
dataset = dataset.astype('float32')
datav = datav.astype('float32')

# Transformation of data
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
datav = scaler.fit_transform(datav)

# Set the 2015-2018 as a traning dataset (1461 days)
train_size = 1461
trainX, testX = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
trainY, testY = datav[0:train_size,:], datav[train_size:len(dataset),:]

#####################################################################

# Set the data as Convlstm input dimension (5 dimensions)
# [samples, timesteps, rows, columns, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1,1,1, 323))
testX = numpy.reshape(testX, (testX.shape[0],1, 1,1, 323))

# Create Convlstm model
model = Sequential()
epochs = 60
model.add(ConvLSTM2D(filters=64, kernel_size=(3,3), activation='relu',
                     padding='same', input_shape=(1,1,1,323)))
model.add(Flatten())
model.add(Dense(323))

# Training begins
model.compile(loss='mse', optimizer='adam')
model.fit(trainX, trainY, epochs=epochs)
model.summary()

# Model training completed, calibration started
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# Inverse standardize the value and restore the value to its original size
testPredict = scaler.inverse_transform(testPredict)

testPredict [testPredict < 0 ] = 0

# All_precipitation is the new data for calibration completion
all_precipitation = testPredict

# Output(precipitation)
all_precipitation = pd.DataFrame(all_precipitation)
# Please revise the path accordingly
all_precipitation.to_csv('/Users/cjchen/Desktop/git-repos/MSWEP_TW/ConvLSTM/output/'+'test_output.csv',header=False, index=False)