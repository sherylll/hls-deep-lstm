from __future__ import print_function
import sys
sys.path += ["."] # Python 3 hack

import numpy as np
import keras
from keras.datasets import mnist
# from keras.models import Sequential
from keras.models import Model, model_from_json
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
    layer = LSTM(hidden_size)(inputs)
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
    for w in weights:
        print(w.shape)
    save_lstm_weights_to_txt('quant_test/cell0',weights[:3],hidden_size)
    save_fc_weights_to_txt('quant_test',weights[3:])

def quantize():
    from models import my_lstm_range, max_abs
    cell_params = load_lstm_weights_from_txt('quant_test/cell0', hidden_size)
    [W_i, W_f, W_c, W_o, U_i, U_f, U_c, U_o, b_i, b_f,b_c,b_o] = cell_params

    Why = np.loadtxt('quant_test/fc/Why.h', delimiter=',')
    by = np.loadtxt('quant_test/fc/by.h', delimiter=',',usecols=range(num_classes)).flatten()

    max_params = [max_abs(p) for p in cell_params]
    max_params += [max_abs(Why), max_abs(by)]
    print(max_params)

    h0,c0={},{}
    h0[-1] = np.zeros(hidden_size)
    c0[-1] = np.zeros(hidden_size)
    max_vals = {"f":np.zeros(4), "i":np.zeros(4), "o":np.zeros(4),"c_act":np.zeros(4),\
        "c_new":np.zeros(3), "h_new":np.zeros(2), "y":np.zeros(2)}
    counter = 0
    for i in range(len(x_test)):
        for t in range(img_cols):
            (h0[t],c0[t])=my_lstm_range(x_test[i][t], h0[t-1], c0[t-1], \
                W_i, W_f, W_c, W_o, U_i, U_f, U_c, U_o,b_i, b_f, b_c, b_o, max_vals)
        Why_h = np.dot(Why,h0[img_cols-1])
        yt = Why_h + by
        max_vals["y"] = np.maximum([max_abs(Why_h),max_abs(yt)], max_vals["y"])

        counter += (np.argmax(yt)==np.argmax(y_test[i]))
    print(counter/10000.0)

    max_vals["params"] = max(max_params)
    return max_vals

def generate_datatype(max_vals):
    min_int_len = 2
    # max_word_len = 16
    frac_len = 8    
    def int_bits(int_val):
        # how many integer bits are necessary
        # for example, 1 or -1 need at least two bits
        return int(np.log2(abs(int_val))) + min_int_len

    def typedef_builder(word_len, int_len, param_name):
        return "typedef ap_fixed<" + str(word_len) + "," + str(int_len) + "> " + param_name + "_t; \n"
    
    def gate_typedef_builder(gate_max_vals, gate_name):
        Wg_x_int = int_bits(gate_max_vals[0])
        Ug_x_int = int_bits(gate_max_vals[1])
        g_in_int = int_bits(gate_max_vals[2])
        g_int = int_bits(gate_max_vals[3])
        typedef_str = typedef_builder(frac_len + Wg_x_int, Wg_x_int, "W" + gate_name + "_x") 
        typedef_str += typedef_builder(frac_len + Ug_x_int, Ug_x_int, "U" + gate_name + "_x") 
        typedef_str += typedef_builder(frac_len + Wg_x_int, g_in_int, gate_name + "_x") 
        typedef_str += typedef_builder(frac_len + Wg_x_int, g_int, gate_name) 
        return typedef_str

    header_file = typedef_builder(frac_len + min_int_len, min_int_len, "data") # input data already normalized

    params_int = int_bits(max_vals["params"])
    header_file += typedef_builder(frac_len + params_int, params_int, "param") 
   
    header_file += gate_typedef_builder(max_vals["i"], "i")
    header_file += gate_typedef_builder(max_vals["f"], "f")
    header_file += gate_typedef_builder(max_vals["o"], "o")
    header_file += gate_typedef_builder(max_vals["c_act"], "c")
    # c_new
    c_new1_int = int_bits(max_vals["c_new"][0])
    c_new2_int = int_bits(max_vals["c_new"][1])
    c_new_int = int_bits(max_vals["c_new"][2])
    header_file += typedef_builder(frac_len + c_new1_int, c_new1_int, "c_new1") 
    header_file += typedef_builder(frac_len + c_new2_int, c_new2_int, "c_new2") 
    header_file += typedef_builder(frac_len + c_new_int, c_new_int, "c_new") 
    # h_new
    c_new_act_int = int_bits(max_vals["h_new"][0])
    h_new_int = int_bits(max_vals["h_new"][1])
    header_file += typedef_builder(frac_len + c_new_act_int, c_new_act_int, "c_new_act") 
    header_file += typedef_builder(frac_len + h_new_int, h_new_int, "h_new") 

    # y
    why_h_int = int_bits(max_vals["y"][0])
    y_int = int_bits(max_vals["y"][1])
    header_file += typedef_builder(frac_len + why_h_int, why_h_int, "why_h") 
    header_file += typedef_builder(frac_len + y_int, y_int, "y") 

    with open("quant_test/parameter.h", "w") as hfile:
        hfile.write(header_file)

def write_good_values(path_test_data, path_written_data):
    # first layer
    [W_i, W_f, W_c, W_o, U_i, U_f, U_c, U_o, b_i, b_f,b_c,b_o] = load_lstm_weights_from_txt('quant_test/cell0', hidden_size)

    Why = np.loadtxt('quant_test/fc/Why.h', delimiter=',')
    by = np.loadtxt('quant_test/fc/by.h', delimiter=',',usecols=range(num_classes)).flatten()

    h0,c0={},{}
    h0[-1] = np.zeros(hidden_size)
    c0[-1] = np.zeros(hidden_size)

    from models import my_lstm
    test_image=np.loadtxt(path_test_data, delimiter=',')
    for t in range(28):
        (h0[t],c0[t])=my_lstm(test_image[t], h0[t-1], c0[t-1], W_i, W_f, W_c, W_o, U_i, U_f, U_c, U_o,b_i, b_f, b_c, b_o)

    yt = np.dot(Why,h0[img_cols-1]) + by
    print(yt)

    # h_dict = {'h0': h0, 'h1':h1}
    with open(path_written_data, 'w') as file:
        file.write('------h0-------\n\n') 
        for k, v in h0.items():
            file.write(str(k) + ' >>> '+ str(v) + '\n\n')         

    # prepare headers
    save_lstm_headers('quant_test/cell0', 0)
    save_fc_headers('quant_test')    

if __name__ == '__main__':
    # train()
    # print(quantize())
    # max_vals = {'f': np.array([1.83959453, 3.28938675, 4.82716574, 1.]), 
    #             'i': np.array([2.70473551, 3.08807776, 3.74156579, 1.]), 
    #             'o': np.array([2.14282328, 4.57429092, 5.37502692, 1.]), 
    #             'c_act': np.array([3.2710689 , 3.53718554, 4.04974074, 0.99939279]), 
    #             'c_new': np.array([10.71407747,  0.99582912, 11.05098167]), 
    #             'h_new': np.array([1., 0.99999998]),
    #             'params': 1.2323434352874756}
    max_vals = quantize()
    generate_datatype(max_vals)
    write_good_values("./test_data/test_digit_4.txt", './quant_test/hidden_states_digit_4_single.txt')

    # Wf_dot_x Uf_dot_h f_in f [1.83959453, 3.28938675, 4.82716574, 1. ]
    # Wi_dot_x Ui_dot_h i_in i [2.70473551, 3.08807776, 3.74156579, 1. ]
    # Wo_dot_x Uo_dot_h o_in o [2.14282328, 4.57429092, 5.37502692, 1. ]
    # Wc_dot_x Uc_dot_h c_in c_act [3.2710689 , 3.53718554, 4.04974074, 0.99939279]
    # c_new_1 c_new_2 c_new c_new_act h_new , [10.71407747,  0.99582912, 11.05098167] [1., 0.99999998]