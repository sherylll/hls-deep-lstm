from __future__ import print_function
import sys
sys.path += ["."] # Python 3 hack

import numpy as np
import keras
from keras.datasets import mnist
# from keras.models import Sequential
from keras.models import Model
from keras.layers import Input, Dense, Bidirectional, LSTM
from keras.layers.recurrent import SimpleRNN
from weight_saver import save_lstm_weights_to_txt, save_fc_weights_to_txt

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

def train(if_header_gen):
    inputs =  Input(shape=(img_rows, img_cols))
    layer = Bidirectional(LSTM(hidden_size))(inputs)
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
    save_lstm_weights_to_txt('bi_lstm_weights/cell0', weights[:3], hidden_size, if_header_gen)
    save_lstm_weights_to_txt('bi_lstm_weights/cell1', weights[3:6], hidden_size, if_header_gen)
    save_fc_weights_to_txt('bi_lstm_weights',weights[6:], if_header_gen)

def test():
    # first layer
    W_i=np.loadtxt('bi_lstm_weights/cell0/W_i.h', delimiter=',')
    W_f=np.loadtxt('bi_lstm_weights/cell0/W_f.h', delimiter=',')
    W_c=np.loadtxt('bi_lstm_weights/cell0/W_c.h', delimiter=',')
    W_o=np.loadtxt('bi_lstm_weights/cell0/W_o.h', delimiter=',')
    U_i=np.loadtxt('bi_lstm_weights/cell0/U_i.h', delimiter=',')
    U_f=np.loadtxt('bi_lstm_weights/cell0/U_f.h', delimiter=',')
    U_c=np.loadtxt('bi_lstm_weights/cell0/U_c.h', delimiter=',')
    U_o=np.loadtxt('bi_lstm_weights/cell0/U_o.h', delimiter=',')
    b_i=np.loadtxt('bi_lstm_weights/cell0/b_i.h', delimiter=',',usecols=range(hidden_size)).flatten()
    b_f=np.loadtxt('bi_lstm_weights/cell0/b_f.h', delimiter=',',usecols=range(hidden_size)).flatten()
    b_c=np.loadtxt('bi_lstm_weights/cell0/b_c.h', delimiter=',',usecols=range(hidden_size)).flatten()
    b_o=np.loadtxt('bi_lstm_weights/cell0/b_o.h', delimiter=',',usecols=range(hidden_size)).flatten()

    # second layer
    W2_i=np.loadtxt('bi_lstm_weights/cell1/W_i.h', delimiter=',')
    W2_f=np.loadtxt('bi_lstm_weights/cell1/W_f.h', delimiter=',')
    W2_c=np.loadtxt('bi_lstm_weights/cell1/W_c.h', delimiter=',')
    W2_o=np.loadtxt('bi_lstm_weights/cell1/W_o.h', delimiter=',')
    U2_i=np.loadtxt('bi_lstm_weights/cell1/U_i.h', delimiter=',')
    U2_f=np.loadtxt('bi_lstm_weights/cell1/U_f.h', delimiter=',')
    U2_c=np.loadtxt('bi_lstm_weights/cell1/U_c.h', delimiter=',')
    U2_o=np.loadtxt('bi_lstm_weights/cell1/U_o.h', delimiter=',')
    b2_i=np.loadtxt('bi_lstm_weights/cell1/b_i.h', delimiter=',',usecols=range(hidden_size)).flatten()
    b2_f=np.loadtxt('bi_lstm_weights/cell1/b_f.h', delimiter=',',usecols=range(hidden_size)).flatten()
    b2_c=np.loadtxt('bi_lstm_weights/cell1/b_c.h', delimiter=',',usecols=range(hidden_size)).flatten()
    b2_o=np.loadtxt('bi_lstm_weights/cell1/b_o.h', delimiter=',',usecols=range(hidden_size)).flatten()

    Why = np.loadtxt('bi_lstm_weights/fc/Why.h', delimiter=',')
    by = np.loadtxt('bi_lstm_weights/fc/by.h', delimiter=',',usecols=range(num_classes)).flatten()

    h0={}
    h1={}
    h0[-1] = np.zeros(hidden_size)
    h1[-1] = np.zeros(hidden_size)

    c0={}
    c1={}
    c0[-1] = np.zeros(hidden_size)
    c1[-1] = np.zeros(hidden_size)

    y={}
    def hard_sigmoid(x):
        return np.clip(0.2 * x + 0.5, 0, 1)

    def my_lstm(x, h, c, Wi, Wf, Wc, Wo, Ui, Uf, Uc, Uo, bi, bf,bc,bo):
        f = hard_sigmoid(np.dot(Wf,x)+np.dot(Uf,h)+bf)
        i = hard_sigmoid(np.dot(Wi,x)+np.dot(Ui,h)+bi)
        o = hard_sigmoid(np.dot(Wo,x)+np.dot(Uo,h)+bo)
        c_new = f * c + i*np.tanh(np.dot(Wc,x)+np.dot(Uc,h)+bc)
        h_new = o * np.tanh(c_new)
        return (h_new, c_new)    

    counter = 0
    for i in range(len(x_test)):
        for t in range(28):
            (h0[t],c0[t])=my_lstm(x_test[i][t], h0[t-1], c0[t-1], W_i, W_f, W_c, W_o, U_i, U_f, U_c, U_o,b_i, b_f, b_c, b_o)
            (h1[t],c1[t])=my_lstm(x_test[i][27-t], h1[t-1], c1[t-1], W2_i, W2_f, W2_c, W2_o, U2_i, U2_f, U2_c, U2_o,b2_i, b2_f, b2_c, b2_o)
        hl_concat=np.concatenate((h0[27], h1[27]))
        yt = np.dot(Why,hl_concat) + by
        counter += (np.argmax(yt)==np.argmax(y_test[i]))

    print(counter/10000.0)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--if_train',type=int)
    parser.add_argument('--if_header_gen',type=int)
    parser.add_argument('--if_test',type=int)
    FLAGS, unparsed = parser.parse_known_args()
    if FLAGS.if_train:
        train(FLAGS.if_header_gen)
    if FLAGS.if_test:
        test()
