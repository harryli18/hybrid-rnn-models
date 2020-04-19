import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import time

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_squared_error
from math import sqrt

import warnings
warnings.filterwarnings('ignore')

def RMSE(y_actual, y_predicted):
    return sqrt(mean_squared_error(y_actual, y_predicted))

# split data in 80%/10%/10% train/validation/test sets
valid_set_size_percentage = 10
test_set_size_percentage = 10

# parameters
seq_len = 100
n_steps = seq_len-1
n_inputs = 11
n_neurons = 2
n_outputs = 11
n_layers = 2
learning_rate = 0.1
batch_size = 100
n_epochs = 50

# use Basic RNN Cell
rnn_layer = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.elu)

# use Basic LSTM Cell
lstm_layer = tf.contrib.rnn.BasicLSTMCell(num_units=n_neurons, activation=tf.nn.elu)

# use Basic GRU cell
gru_layer = tf.contrib.rnn.GRUCell(num_units=n_neurons, activation=tf.nn.leaky_relu)

# use LSTM Cell with peephole connections
#layers = [tf.contrib.rnn.LSTMCell(num_units=n_neurons,
#                                  activation=tf.nn.leaky_relu, use_peepholes = True)
#          for layer in range(n_layers)]

layers = [tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.elu)
          for layer in range(n_layers)]


GRU = tf.contrib.rnn.MultiRNNCell(cells=[gru_layer])
LSTM = tf.contrib.rnn.MultiRNNCell(cells=[lstm_layer])

GRU_GRU =  tf.contrib.rnn.MultiRNNCell(cells=[gru_layer,lstm_layer])
GRU_LSTM = tf.contrib.rnn.MultiRNNCell(cells=[gru_layer, lstm_layer])
LSTM_GRU = tf.contrib.rnn.MultiRNNCell(cells=[lstm_layer,gru_layer])
LSTM_LSTM = tf.contrib.rnn.MultiRNNCell(cells=[lstm_layer,lstm_layer])

RnnModelDict = {'LSTM': LSTM, 'GRU': GRU, 'LSTM_LSTM': LSTM_LSTM, 'GRU_GRU': GRU_GRU, 'LSTM_GRU': LSTM_GRU,
                'GRU_LSTM': GRU_LSTM}

# function to create train, validation, test data given stock data and sequence length
# the training sets are the sequences (20)
# this is the methods of time series prediction
def load_data(data_raw, seq_len):
    #     data_raw = stock.as_matrix() # convert to numpy array
    data = []

    # create all possible sequences of length seq_len
    for index in range(len(data_raw) - seq_len):
        data.append(data_raw[index: index + seq_len])

    data = np.array(data);
    valid_set_size = int(np.round(valid_set_size_percentage / 100 * data.shape[0]));
    test_set_size = int(np.round(test_set_size_percentage / 100 * data.shape[0]));
    train_set_size = data.shape[0] - (valid_set_size + test_set_size);

    x_train = data[:train_set_size, :-1, :]
    y_train = data[:train_set_size, -1, :]

    x_valid = data[train_set_size:train_set_size + valid_set_size, :-1, :]
    y_valid = data[train_set_size:train_set_size + valid_set_size, -1, :]

    x_test = data[train_set_size + valid_set_size:, :-1, :]
    y_test = data[train_set_size + valid_set_size:, -1, :]

    return [x_train, y_train, x_valid, y_valid, x_test, y_test]

## Basic Cell RNN in tensorflow
def get_next_batch(batch_size):
    global index_in_epoch, x_train, perm_array
    start = index_in_epoch
    index_in_epoch += batch_size

    if index_in_epoch > x_train.shape[0]:
        np.random.shuffle(perm_array)  # shuffle permutation array
        start = 0  # start next epoch
        index_in_epoch = batch_size

    end = index_in_epoch
    return x_train[perm_array[start:end]], y_train[perm_array[start:end]]


def run_rnn_model(key):


    train_set_size = x_train.shape[0]
    test_set_size = x_test.shape[0]

    tf.reset_default_graph()
    # In TensorFlowterminology, we then feed data into the graph through these placeholders.
    X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
    y = tf.placeholder(tf.float32, [None, n_outputs])

    multi_layer_cell = RnnModelDict[key]
    rnn_outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)

    stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, n_neurons])
    stacked_outputs = tf.layers.dense(stacked_rnn_outputs, n_outputs)
    outputs = tf.reshape(stacked_outputs, [-1, n_steps, n_outputs])
    outputs = outputs[:, n_steps - 1, :]  # keep only last output of sequence

    loss = tf.reduce_mean(tf.square(outputs - y))  # loss function = mean squared error
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss)

    # run graph
    start = time.process_time()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for iteration in range(int(n_epochs * train_set_size / batch_size)):
            x_batch, y_batch = get_next_batch(batch_size)  # fetch the next training batch
            sess.run(training_op, feed_dict={X: x_batch, y: y_batch})
            if iteration % int(5 * train_set_size / batch_size) == 0:
                mse_train = loss.eval(feed_dict={X: x_train, y: y_train})
                mse_valid = loss.eval(feed_dict={X: x_valid, y: y_valid})
                print('%.2f epochs: MSE train/valid = %.6f/%.6f' % (
                    iteration * batch_size / train_set_size, mse_train, mse_valid))

        y_train_pred = sess.run(outputs, feed_dict={X: x_train})
        y_valid_pred = sess.run(outputs, feed_dict={X: x_valid})
        y_test_pred = sess.run(outputs, feed_dict={X: x_test})

    print('time taken for {} model traning: {} for epoch: {}, n_neurons: {}, batch_size: {}, learning_rate: {}, '
          'n_steps: {}'.format(key, time.process_time() - start, n_epochs, n_neurons, batch_size, learning_rate, n_steps))

    print('training error for', key)
    print('mean_squared_error', mean_squared_error(y_train[:, 4], y_train_pred[:, 4]))
    print('r2_score', r2_score(y_train[:, 4], y_train_pred[:, 4]))
    print('RMSE', RMSE(y_train[:, 4], y_train_pred[:, 4]))

    print('validation error for', key)
    print('mean_squared_error', mean_squared_error(y_valid[:, 4], y_valid_pred[:, 4]))
    print('r2_score', r2_score(y_valid[:, 4], y_valid_pred[:, 4]))
    print('RMSE', RMSE(y_valid[:, 4], y_valid_pred[:, 4]))

    print('test error for', key)
    print('mean_squared_error', mean_squared_error(y_test[:, 4], y_test_pred[:, 4]))
    print('r2_score', r2_score(y_test[:, 4], y_test_pred[:, 4]))
    print('RMSE', RMSE(y_test[:, 4], y_test_pred[:, 4]))
    return('model training complete for:', key)

if __name__ == "__main__":

    num_data_df = pd.read_csv('data_slim_subset.csv')
    standardScaler = StandardScaler()
    num_data_scaled_df = standardScaler.fit_transform(num_data_df)

    x_train, y_train, x_valid, y_valid, x_test, y_test = load_data(num_data_scaled_df, seq_len)

    index_in_epoch = 0
    perm_array = np.arange(x_train.shape[0])
    np.random.shuffle(perm_array)


    for key in RnnModelDict:
        run_rnn_model(key)

    ft_list = []
    for j in range(15):
        ft_list.append([j, num_data_df.columns[j]])
    print(ft_list)
    ft = 4  # 4 PM2.5

    ## show predictions
    plt.figure(figsize=(35, 5));
    plt.subplot(1, 2, 1);

    plt.plot(np.arange(y_train.shape[0]), y_train[:, ft], color='blue', label='train target')

    plt.plot(np.arange(y_train.shape[0], y_train.shape[0] + y_valid.shape[0]), y_valid[:, ft],
             color='gray', label='valid target')

    plt.plot(np.arange(y_train.shape[0] + y_valid.shape[0],
                       y_train.shape[0] + y_test.shape[0] + y_test.shape[0]),
             y_test[:, ft], color='black', label='test target')

    plt.plot(np.arange(y_train_pred.shape[0]), y_train_pred[:, ft], color='red',
             label='test prediction')

    plt.plot(np.arange(y_train_pred.shape[0], y_train_pred.shape[0] + y_valid_pred.shape[0]),
             y_valid_pred[:, ft], color='orange', label='valid prediction')

    plt.plot(np.arange(y_train_pred.shape[0] + y_valid_pred.shape[0],
                       y_train_pred.shape[0] + y_valid_pred.shape[0] + y_test_pred.shape[0]),
             y_test_pred[:, ft], color='green', label='test prediction')

    plt.title('past and future PM2.5 Level')
    plt.xlabel('time [days]')
    plt.ylabel('normalized PM2.5 Level')
    plt.legend(loc='best')



