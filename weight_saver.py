import numpy as np

def prepare_c_header(filename, macro_name):
    new_header = '#define ' + str(macro_name) + ' {'
    with open(filename) as f:
        lines = f.readlines()
    if len(lines) == 1: # bias
        new_header += str(lines[0])
    else: # weights
        for line in lines:
            new_header += '{' + line.strip('\n').strip('\r').strip('\rn') + '},' 
    new_header += '}'

    with open(filename, 'w+') as f:
        f.write(new_header)

def save_lstm_headers(path, cell_number=0):        
    print('generating headers...')
    cell_num_str = ''
    if cell_number:
        cell_num_str = str(cell_number)
    prepare_c_header(path+'/W_i.h', 'W'+cell_num_str+'_I')
    prepare_c_header(path+'/W_f.h', 'W'+cell_num_str+'_F')
    prepare_c_header(path+'/W_c.h', 'W'+cell_num_str+'_C')
    prepare_c_header(path+'/W_o.h', 'W'+cell_num_str+'_O')

    prepare_c_header(path+'/U_i.h', 'U'+cell_num_str+'_I')
    prepare_c_header(path+'/U_f.h', 'U'+cell_num_str+'_F')
    prepare_c_header(path+'/U_c.h', 'U'+cell_num_str+'_C')
    prepare_c_header(path+'/U_o.h', 'U'+cell_num_str+'_O')

    prepare_c_header(path+'/b_i.h', 'B'+cell_num_str+'_I')
    prepare_c_header(path+'/b_f.h', 'B'+cell_num_str+'_F')
    prepare_c_header(path+'/b_c.h', 'B'+cell_num_str+'_C')
    prepare_c_header(path+'/b_o.h', 'B'+cell_num_str+'_O')

def save_lstm_weights_to_txt(path, weights, hidden_size, if_header_gen=False):
    # save keras parameters as file
    W_ifo = np.transpose(weights[0])
    W_i = W_ifo[:hidden_size]
    W_f = W_ifo[hidden_size:2*hidden_size]
    W_c = W_ifo[hidden_size*2:hidden_size*3]
    W_o = W_ifo[hidden_size*3:]

    np.savetxt(path+'/W_i.h', W_i, delimiter=',')
    np.savetxt(path+'/W_f.h', W_f, delimiter=',')
    np.savetxt(path+'/W_c.h', W_c, delimiter=',')
    np.savetxt(path+'/W_o.h', W_o, delimiter=',')
    
    U_ifo = np.transpose(weights[1])
    U_i = U_ifo[:hidden_size]
    U_f = U_ifo[hidden_size:2*hidden_size]
    U_c = U_ifo[hidden_size*2:hidden_size*3]
    U_o = U_ifo[hidden_size*3:]
    np.savetxt(path+'/U_i.h', U_i, delimiter=',')
    np.savetxt(path+'/U_f.h', U_f, delimiter=',')
    np.savetxt(path+'/U_c.h', U_c, delimiter=',')
    np.savetxt(path+'/U_o.h', U_o, delimiter=',')    

    bg = weights[2]
    b_i = bg[:hidden_size]
    b_f = bg[hidden_size:2*hidden_size]
    b_c = bg[hidden_size*2:hidden_size*3]
    b_o = bg[hidden_size*3:]
    np.savetxt(path+'/b_i.h', b_i, newline=',')
    np.savetxt(path+'/b_f.h', b_f, newline=',')
    np.savetxt(path+'/b_c.h', b_c, newline=',')
    np.savetxt(path+'/b_o.h', b_o, newline=',')

    if if_header_gen:
        save_lstm_headers(path)

def load_lstm_weights_from_txt(prefix, hidden_size):
    W_i=np.loadtxt(prefix+'/W_i.h', delimiter=',')
    W_f=np.loadtxt(prefix+'/W_f.h', delimiter=',')
    W_c=np.loadtxt(prefix+'/W_c.h', delimiter=',')
    W_o=np.loadtxt(prefix+'/W_o.h', delimiter=',')
    U_i=np.loadtxt(prefix+'/U_i.h', delimiter=',')
    U_f=np.loadtxt(prefix+'/U_f.h', delimiter=',')
    U_c=np.loadtxt(prefix+'/U_c.h', delimiter=',')
    U_o=np.loadtxt(prefix+'/U_o.h', delimiter=',')
    b_i=np.loadtxt(prefix+'/b_i.h', delimiter=',',usecols=range(hidden_size)).flatten()
    b_f=np.loadtxt(prefix+'/b_f.h', delimiter=',',usecols=range(hidden_size)).flatten()
    b_c=np.loadtxt(prefix+'/b_c.h', delimiter=',',usecols=range(hidden_size)).flatten()
    b_o=np.loadtxt(prefix+'/b_o.h', delimiter=',',usecols=range(hidden_size)).flatten()
    return [W_i, W_f, W_c, W_o, U_i, U_f, U_c, U_o, b_i, b_f,b_c,b_o]

def save_fc_headers(path):
    prepare_c_header(path+'/fc/Why.h', 'WHY')
    prepare_c_header(path+'/fc/by.h', 'BY')

def save_fc_weights_to_txt(path, weights, if_header_gen=False):
    np.savetxt(path+'/fc/Why.h', np.transpose(weights[0]), delimiter=',')
    np.savetxt(path+'/fc/by.h', weights[1], newline=',') 
    if if_header_gen:
        save_fc_headers(path)