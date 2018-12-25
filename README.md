# lstm-for-time-series

import numpy as np
import pandas as pd
import sklearn
import sklearn.preprocessing
from sklearn.metrics import mean_squared_error
import datetime
import tensorflow as tf
import math
from math import sqrt
import statsmodels
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Input, Dense, GRU, Embedding
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau


#hong kong 15 mins intraday data
hkg_data = pd.read_csv('HKG.IDXHKD_Candlestick_15_M_BID_18.11.2013-24.11.2018.csv') 

#data manipulation for time column
updated_time = []
for line in hkg_data['Local time']:
    line = line[:-13]
    updated_time.append(line)
hkg_data['obj_time'] = updated_time

#obj_time will be used for time
hkg_data.drop(['Local time'],axis = 1)

hkg_data['obj_time'] = pd.to_datetime(hkg_data['obj_time'], format='%d.%m.%Y %H:%M:%S')
hkg_data_row = [row for index, row in hkg_data.iterrows()]

indexed_hkg_data_df = pd.DataFrame(hkg_data_row,index=hkg_data['obj_time'])

#data preperation for lstm

target_column = ['Close']
shift_time = 1
target_df = hkg_df[target_column].shift(-shift_time)

#converts to numpy arrays
x_data = hkg_df.values[:-shift_time]
y_data = target_df.values[:-shift_time]


num_data = len(x_data)
num_train = int(train_split * num_data)
num_test = num_data - num_train


x_train = x_data[0:num_train]
x_test = x_data[num_train:]

y_train = y_data[0:num_train]
y_test = y_data[num_train:]

num_x_signals = x_data.shape[1]

num_y_signals = y_data.shape[1]

#scaling the values 
scaler = MinMaxScaler()

x_train_scaled = x_scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

y_train_scaled = scaler.fit_transform(y_train)
y_test_scaled = scaler.transform(y_test)


def batch_generator(batch_size, sequence_length):
    """
    Generator function for creating random batches of training-data.
    """

    # Infinite loop.
    while True:
        # Allocate a new array for the batch of input-signals.
        x_shape = (batch_size, sequence_length, num_x_signals)
        x_batch = np.zeros(shape=x_shape, dtype=np.float16)

        # Allocate a new array for the batch of output-signals.
        y_shape = (batch_size, sequence_length, num_y_signals)
        y_batch = np.zeros(shape=y_shape, dtype=np.float16)

        # Fill the batch with random sequences of data.
        for i in range(batch_size):
            # Get a random start-index.
            # This points somewhere into the training-data.
            idx = np.random.randint(num_train - sequence_length)
            
            # Copy the sequences of data starting at this index.
            x_batch[i] = x_train_scaled[idx:idx+sequence_length]
            y_batch[i] = y_train_scaled[idx:idx+sequence_length]
        
        yield (x_batch, y_batch)

sequence_length = 24 * 5 * 8 # 2 ay
batch_size = 256
generator = batch_generator(batch_size=batch_size,
                            sequence_length=sequence_length)
                            
x_batch, y_batch = next(generator)

validation_data = (np.expand_dims(x_test_scaled, axis=0),
                   np.expand_dims(y_test_scaled, axis=0))

model = Sequential()
model.add(GRU(units=512,
              return_sequences=True,
              input_shape=(None, num_x_signals,)))

if False:
    from tensorflow.python.keras.initializers import RandomUniform

    # Maybe use lower init-ranges.
    init = RandomUniform(minval=-0.05, maxval=0.05)

    model.add(Dense(num_y_signals,
                    activation='linear',
                    kernel_initializer=init))
          
step_num = 50

def loss_mse_warmup(y_true, y_pred):
   
    y_true_slice = y_true[:, step_num:, :]
    y_pred_slice = y_pred[:, step_num:, :]

    loss = tf.losses.mean_squared_error(labels=y_true_slice,
                                        predictions=y_pred_slice)
    loss_mean = tf.reduce_mean(loss)
    return loss_mean
                   

optimizer = RMSprop(lr=1e-3)
model.compile(loss=loss_mse_warmup, optimizer=optimizer)


checkpoint = 'bitirme.keras'
model_checkpoint = ModelCheckpoint(filepath=checkpoint,
                                      monitor='val_loss',
                                      verbose=1,
                                      save_weights_only=True,
                                      save_best_only=True)

early_stopping = EarlyStopping(monitor='val_loss',
                                        patience=5, verbose=1)

tensorboard = TensorBoard(log_dir='./bitirme/',
                                   histogram_freq=0,
                                   write_graph=False)
 
reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                       factor=0.1,
                                       min_lr=1e-4,
                                       patience=0,
                                       verbose=1)
                                       
callbacks = [early_stopping,
             model_checkpoint,
             tensorboard,
             reduce_lr]
             
%%time
model.fit_generator(generator=generator,
                    epochs=20,
                    steps_per_epoch=100,
                    validation_data=validation_data,
                    callbacks=callbacks)           
