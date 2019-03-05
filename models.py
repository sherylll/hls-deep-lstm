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

def max_abs(arr):
    return max(abs(np.amax(arr)),abs(np.amin(arr)))

def my_lstm_range(x, h, c, Wi, Wf, Wc, Wo, Ui, Uf, Uc, Uo, bi, bf,bc,bo, max_vals={}):
    #  there are 21 intermediate values
    # Wf_dot_x Uf_dot_h f_in f
    # Wi_dot_x Ui_dot_h i_in i
    # Wo_dot_x Uo_dot_h o_in o
    # Wc_dot_x Uc_dot_h c_in c_act
    # c_new_1 c_new_2 c_new c_new_act h_new
    # max_vals should be a dict

    Wf_dot_x = np.dot(Wf,x)
    Uf_dot_h = np.dot(Uf,h)
    f_in = Wf_dot_x + Uf_dot_h + bf
    f = hard_sigmoid(f_in)
    f_max_old = max_vals["f"]
    f_max_curr = [max_abs(Wf_dot_x), max_abs(Uf_dot_h), max_abs(f_in), max_abs(f)]
    max_vals["f"] = np.maximum(f_max_curr, f_max_old)

    Wi_dot_x = np.dot(Wi,x)
    Ui_dot_h = np.dot(Ui,h)
    i_in = Wi_dot_x + Ui_dot_h + bi
    i = hard_sigmoid(i_in)
    i_max_old = max_vals["i"] 
    i_max_curr = [max_abs(Wi_dot_x), max_abs(Ui_dot_h), max_abs(i_in), max_abs(i)]
    max_vals["i"] = np.maximum(i_max_curr, i_max_old)

    Wo_dot_x = np.dot(Wo,x)
    Uo_dot_h = np.dot(Uo,h)
    o_in = Wo_dot_x + Uo_dot_h + bo
    o = hard_sigmoid(o_in)
    o_max_old = max_vals["o"] 
    o_max_curr = [max_abs(Wo_dot_x), max_abs(Uo_dot_h), max_abs(o_in), max_abs(o)]
    max_vals["o"] = np.maximum(o_max_curr, o_max_old)

    Wc_dot_x = np.dot(Wc,x)
    Uc_dot_h = np.dot(Uc,h)
    c_in = Wc_dot_x + Uc_dot_h + bc
    c_act = np.tanh(c_in)
    c_max_old = max_vals["c_act"] 
    c_max_curr = [max_abs(Wc_dot_x), max_abs(Uc_dot_h), max_abs(c_in), max_abs(c_act)]
    max_vals["c_act"] = np.maximum(c_max_curr, c_max_old)
    
    # c_new = f * c + i*np.tanh(np.dot(Wc,x)+np.dot(Uc,h)+bc)
    c_new_1 = f * c
    c_new_2 = i * c_act
    c_new = c_new_2 + c_new_1
    c_new_max_old = max_vals["c_new"] 
    c_new_max_curr = [max_abs(c_new_1),max_abs(c_new_2),max_abs(c_new)]
    max_vals["c_new"] = np.maximum(c_new_max_old, c_new_max_curr)

    c_new_act = np.tanh(c_new)
    h_new = o * c_new_act
    h_new_max_old = max_vals["h_new"] 
    h_new_max_curr = [max_abs(c_new_act), max_abs(h_new)]
    max_vals["h_new"] = np.maximum(h_new_max_old, h_new_max_curr)

    return (h_new, c_new)        