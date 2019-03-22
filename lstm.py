#!/usr/bin/env python
import pandas as pd
import numpy as np
import sys

from tensorflow.python.keras.models import Sequential  # , load_model, save_model
from tensorflow.python.keras.layers import Dense, LSTM, Dropout
from tensorflow.python.keras.callbacks import TensorBoard

from collections import deque
# import matplotlib.pyplot as plt
import random
from time import time


# Load csv file into a dataframe, set the index to be the date, drop original id number
df = pd.DataFrame()
if (len(sys.argv) == 1):
    df = pd.read_csv("arenas.csv", parse_dates=['date'])
else:
    df = pd.read_csv(sys.argv[1], parse_dates=['date'])

df.drop('id', axis=1, inplace=True)
df.set_index('date', inplace=True)

# Convert date to date time format
df.index = pd.to_datetime(df.index)
# Generate a mapping of the venues to integers
venues = df["venue"].unique().tolist()
venueToId = {venues[i]:i for i in range(len(venues))}
idToVenue = {i:venues[i] for i in range(len(venues))}

# Formatting data for input to network
df['venue'] = df['venue'].map(venueToId)
df['temperature'] = df['temperature'].map(lambda temp : temp / 300) # normalize temperatures

# Arrange rows in chronological order
df.sort_values(by=['date'], inplace=True)

# Aggregates the data for each venue {'venue1': [venue, temp, precip], 'venue2':[venue, temp, precip]}
# Resamples the data to be daily measurements. Interpolates the data to fill in the missing measurements
venue_data = {}
for venue in venues:
    v_df = df.loc[df['venue'] == venueToId[venue]].loc[:,['venue', 'temperature', 'precipitation']]
    # v_df = v_df[~v_df.index.duplicated()]
    v_df = v_df.resample('D').mean()
    v_df = v_df.interpolate(method='linear')
    venue_data[venue] = v_df
    # v_df['temperature'].plot()
    # plt.title(venue)
    # plt.show()
    # print(v_df)

SEQ_LEN = 20
sequences = []

# Creates a list of sequential data of the form ([[venue, temp0], [venue,temp1]], expected_next_temperature)
# Ignores precipitation since there is not enough data (lots of zeroes)
for venue in venues:
    prev_seq = deque(maxlen=SEQ_LEN)
    data = venue_data[venue].iloc[:,:]
    for i in data.iterrows():
        # date = int(i[1]['date'])
        temp = float(i[1]['temperature'])
        # precip = float(i[1]['precipitation'])

        # With previous weather data, predict this next temp
        if (len(prev_seq) == SEQ_LEN):
            sequences.append([np.array(prev_seq), temp])

        # Add temp to the sequence, remove old ones if necessary
        prev_seq.append((venueToId[venue], temp))

# Randomize sequence ordering to learn from different times/venues
random.shuffle(sequences)

x = []
y = []
for seq, target in sequences:
    x.append(seq)
    y.append([target])

# Create training set
training_size = int(len(x)*0.9)
x_train = np.array(x[: training_size])
y_train = np.array(y[:training_size])

# Create test set
x_test = np.array(x[training_size:])
y_test = np.array(y[training_size:])

# Generate sequence from the last 20 real temperatures for Capital Center to predict the next temperature
cap_center_data = df.loc[df['venue'] == venueToId['Capital Center']]
start = cap_center_data.shape[0] - SEQ_LEN
end = start + SEQ_LEN
cap_center_data = cap_center_data.iloc[start:end, 0:3]

cc_sequence = []
for row in cap_center_data.iterrows():
    temp = float(row[1]['temperature'])
    cc_sequence.append((venueToId['Capital Center'], temp))
    # With previous weather data, predict this next temp
cc_sequence = np.array(cc_sequence).reshape(1, SEQ_LEN, 2)

# Sequential model with 3 LSTM layers which each feed into a dropout layer.
# LSTM -> Dropout Layer -> ... -> Dense -> Dropout -> Dense (output layer)
# LSTM layer to take in sequences of data
# Dropout layer to prevent over-fitting. Ignores a number of nuerons to force network to learn more robust features
BATCH_SIZE = 128    # Number of sequences per batch
MAX_EPOCHS = 10     # Max number of training cycles
n_lstm = 128        # Number of hidden unints in LSTM

model = Sequential()
model.add(LSTM(n_lstm, input_shape=(x_train.shape[1:]), activation='tanh', return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(n_lstm, activation='tanh', return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(n_lstm, activation='tanh'))
model.add(Dropout(0.2))

model.add(Dense(32, activation='tanh'))
model.add(Dropout(0.2))

model.add(Dense(1, activation='tanh'))

# Configure the model to use mean square error and the adam optimizer.
# Does not call metric=['accuracy'] since that is used for categorical data
model.compile(loss='mse',
              optimizer='adam',
              metrics=['mse'])

# Log data at intervals during training
tensorboard = TensorBoard(log_dir='logs/{}_{}_{}_{}.hd5'.format(MAX_EPOCHS, BATCH_SIZE, n_lstm, time()))


history = model.fit(x_train, y_train,
                    batch_size=BATCH_SIZE,
                    epochs=MAX_EPOCHS,
                    validation_data=(x_test, y_test),
                    callbacks=[tensorboard])


# prediction = model.predict(x_test)

# Save model
# save_model(model, "models/{}_{}_{}_{}.hd5".format(MAX_EPOCHS, BATCH_SIZE, n_lstm, time()), include_optimizer=True)

# for i in range(len(prediction)):
#     print(y_test[i] * 300, prediction[i]*300)

# Predict the next day based on the last 20 real temperatures at the Capital center
prediction = model.predict(cc_sequence)
print("The temperature at the Capital Center will be: ", float(prediction[0]*300),"F")
