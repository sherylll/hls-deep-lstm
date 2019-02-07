from __future__ import print_function
import sys
sys.path += ["."] # Python 3 hack

import numpy as np
import keras
from keras.datasets import mnist
# from keras.models import Sequential
from keras.models import Model
from keras.layers import Input, Dense, LSTM
from weight_saver import *

batch_size = 64
num_classes = 10
epochs = 1
hidden_size = 128
# input image dimensions
img_rows, img_cols = 28, 28
# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

def train():
    inputs =  Input(shape=(img_rows, img_cols))
    layer = LSTM(hidden_size, return_sequences=True)(inputs)
    layer = LSTM(hidden_size, return_sequences=True)(layer)
    layer = LSTM(hidden_size, return_sequences=True)(layer)
    layer = LSTM(hidden_size)(layer)
    predictions = Dense(num_classes, activation='softmax')(layer)

    model = Model(inputs=inputs, outputs=predictions)

    model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adadelta(),
                metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=[x_test, y_test])

    weights = model.get_weights()
    # for w in weights:
    #     print(w.shape)
    save_lstm_weights_to_txt('stacked_lstm_weights/cell0', weights[:3], hidden_size)
    save_lstm_weights_to_txt('stacked_lstm_weights/cell1', weights[3:6], hidden_size)
    save_lstm_weights_to_txt('stacked_lstm_weights/cell2', weights[6:9], hidden_size)
    save_lstm_weights_to_txt('stacked_lstm_weights/cell3', weights[9:12], hidden_size)
    save_fc_weights_to_txt('stacked_lstm_weights',weights[12:])

def test():
    # first layer
    [W_i, W_f, W_c, W_o, U_i, U_f, U_c, U_o, b_i, b_f,b_c,b_o] = load_lstm_weights_from_txt('stacked_lstm_weights/cell0', hidden_size)
    # second layer
    [W1_i, W1_f, W1_c, W1_o, U1_i, U1_f, U1_c, U1_o, b1_i, b1_f,b1_c,b1_o] = load_lstm_weights_from_txt('stacked_lstm_weights/cell1', hidden_size)
    [W2_i, W2_f, W2_c, W2_o, U2_i, U2_f, U2_c, U2_o, b2_i, b2_f,b2_c,b2_o] = load_lstm_weights_from_txt('stacked_lstm_weights/cell2', hidden_size)
    [W3_i, W3_f, W3_c, W3_o, U3_i, U3_f, U3_c, U3_o, b3_i, b3_f,b3_c,b3_o] = load_lstm_weights_from_txt('stacked_lstm_weights/cell3', hidden_size)

    Why = np.loadtxt('stacked_lstm_weights/fc/Why.h', delimiter=',')
    by = np.loadtxt('stacked_lstm_weights/fc/by.h', delimiter=',',usecols=range(num_classes)).flatten()

    h0, h1, h2, h3={}, {}, {}, {}
    h0[-1] = np.zeros(hidden_size)
    h1[-1] = np.zeros(hidden_size)
    h2[-1] = np.zeros(hidden_size)
    h3[-1] = np.zeros(hidden_size)

    c0,c1,c2,c3={}, {}, {}, {}
    c0[-1] = np.zeros(hidden_size)
    c1[-1] = np.zeros(hidden_size)
    c2[-1] = np.zeros(hidden_size)
    c3[-1] = np.zeros(hidden_size)

    from models import my_lstm
    counter = 0
    for i in range(len(x_test)):
        for t in range(img_cols):
            (h0[t],c0[t])=my_lstm(x_test[i][t], h0[t-1], c0[t-1], W_i, W_f, W_c, W_o, U_i, U_f, U_c, U_o,b_i, b_f, b_c, b_o)
            (h1[t],c1[t])=my_lstm(h0[t], h1[t-1], c1[t-1], W1_i, W1_f, W1_c, W1_o, U1_i, U1_f, U1_c, U1_o, b1_i, b1_f, b1_c, b1_o)
            (h2[t],c2[t])=my_lstm(h1[t], h2[t-1], c2[t-1], W2_i, W2_f, W2_c, W2_o, U2_i, U2_f, U2_c, U2_o, b2_i, b2_f, b2_c, b2_o)
            (h3[t],c3[t])=my_lstm(h2[t], h3[t-1], c3[t-1], W3_i, W3_f, W3_c, W3_o, U3_i, U3_f, U3_c, U3_o, b3_i, b3_f, b3_c, b3_o)

        yt = np.dot(Why,h3[img_cols-1]) + by
        counter += (np.argmax(yt)==np.argmax(y_test[i]))

    print(counter/10000.0)

def write_good_values(path_test_data, path_written_data):
    # first layer
    [W_i, W_f, W_c, W_o, U_i, U_f, U_c, U_o, b_i, b_f,b_c,b_o] = load_lstm_weights_from_txt('stacked_lstm_weights/cell0', hidden_size)
    # second layer
    [W1_i, W1_f, W1_c, W1_o, U1_i, U1_f, U1_c, U1_o, b1_i, b1_f,b1_c,b1_o] = load_lstm_weights_from_txt('stacked_lstm_weights/cell1', hidden_size)
    [W2_i, W2_f, W2_c, W2_o, U2_i, U2_f, U2_c, U2_o, b2_i, b2_f,b2_c,b2_o] = load_lstm_weights_from_txt('stacked_lstm_weights/cell2', hidden_size)
    [W3_i, W3_f, W3_c, W3_o, U3_i, U3_f, U3_c, U3_o, b3_i, b3_f,b3_c,b3_o] = load_lstm_weights_from_txt('stacked_lstm_weights/cell3', hidden_size)

    Why = np.loadtxt('stacked_lstm_weights/fc/Why.h', delimiter=',')
    by = np.loadtxt('stacked_lstm_weights/fc/by.h', delimiter=',',usecols=range(num_classes)).flatten()

    h0, h1, h2, h3={}, {}, {}, {}
    h0[-1] = np.zeros(hidden_size)
    h1[-1] = np.zeros(hidden_size)
    h2[-1] = np.zeros(hidden_size)
    h3[-1] = np.zeros(hidden_size)

    c0,c1,c2,c3={}, {}, {}, {}
    c0[-1] = np.zeros(hidden_size)
    c1[-1] = np.zeros(hidden_size)
    c2[-1] = np.zeros(hidden_size)
    c3[-1] = np.zeros(hidden_size)

    from models import my_lstm
    test_image=np.loadtxt(path_test_data, delimiter=',')
    for t in range(28):
        (h0[t],c0[t])=my_lstm(test_image[t], h0[t-1], c0[t-1], W_i, W_f, W_c, W_o, U_i, U_f, U_c, U_o,b_i, b_f, b_c, b_o)
        (h1[t],c1[t])=my_lstm(h0[t], h1[t-1], c1[t-1], W1_i, W1_f, W1_c, W1_o, U1_i, U1_f, U1_c, U1_o, b1_i, b1_f, b1_c, b1_o)
        (h2[t],c2[t])=my_lstm(h1[t], h2[t-1], c2[t-1], W2_i, W2_f, W2_c, W2_o, U2_i, U2_f, U2_c, U2_o, b2_i, b2_f, b2_c, b2_o)
        (h3[t],c3[t])=my_lstm(h2[t], h3[t-1], c3[t-1], W3_i, W3_f, W3_c, W3_o, U3_i, U3_f, U3_c, U3_o, b3_i, b3_f, b3_c, b3_o)

    yt = np.dot(Why,h3[img_cols-1]) + by
    print(yt)

    # h_dict = {'h0': h0, 'h1':h1}
    with open(path_written_data, 'w') as file:
        file.write('------h0-------\n\n') 
        for k, v in h0.items():
            file.write(str(k) + ' >>> '+ str(v) + '\n\n')
        file.write('------h1-------\n\n')     
        for k, v in h1.items():
            file.write(str(k) + ' >>> '+ str(v) + '\n\n')   
        file.write('------h2-------\n\n')     
        for k, v in h2.items():
            file.write(str(k) + ' >>> '+ str(v) + '\n\n')   
        file.write('------h3-------\n\n')     
        for k, v in h3.items():
            file.write(str(k) + ' >>> '+ str(v) + '\n\n')                   

    # prepare headers
    save_lstm_headers('stacked_lstm_weights/cell0', 0)
    save_lstm_headers('stacked_lstm_weights/cell1', 1)
    save_lstm_headers('stacked_lstm_weights/cell2', 2)
    save_lstm_headers('stacked_lstm_weights/cell3', 3)    
    save_fc_headers('stacked_lstm_weights')    

if __name__ == '__main__':
    # train, test, save reference data & prepare headers
    # python blstm.py --if_train --if_test --if_states_gen 1
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--if_train',type=int)
    parser.add_argument('--if_test',type=int)
    parser.add_argument('--if_states_gen',type=int)

    FLAGS, unparsed = parser.parse_known_args()
    if FLAGS.if_train:
        train()
    if FLAGS.if_test:
        test()
    if FLAGS.if_states_gen:
        write_good_values('./test_data/test_digit_4.txt', './test_data/hidden_states_digit_4_stacked.txt')