import numpy as np

def hard_sigmoid(x):
    return np.clip(0.2 * x + 0.5, 0, 1)

def my_lstm(x, h, c, Wi, Wf, Wc, Wo, Ui, Uf, Uc, Uo, bi, bf,bc,bo):
    f = hard_sigmoid(np.dot(Wf,x)+np.dot(Uf,h)+bf)
    i = hard_sigmoid(np.dot(Wi,x)+np.dot(Ui,h)+bi)
    o = hard_sigmoid(np.dot(Wo,x)+np.dot(Uo,h)+bo)
    c_new = f * c + i*np.tanh(np.dot(Wc,x)+np.dot(Uc,h)+bc)
    h_new = o * np.tanh(c_new)
    return (h_new, c_new)    