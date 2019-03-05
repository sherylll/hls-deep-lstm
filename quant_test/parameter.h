#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include "ap_fixed.h"
#include "../../hlslib/nn_utils/nn_common.h"
#include "../../hlslib/nn_utils/nn_recurrent.h"

#include "cell0/W_i.h"
#include "cell0/W_f.h"
#include "cell0/W_c.h"
#include "cell0/W_o.h"
#include "cell0/U_i.h"
#include "cell0/U_f.h"
#include "cell0/U_c.h"
#include "cell0/U_o.h"
#include "cell0/b_i.h"
#include "cell0/b_f.h"
#include "cell0/b_c.h"
#include "cell0/b_o.h"

#include "fc/by.h"
#include "fc/Why.h"

// using fixed only with caution! accuracy w/o quant is very low!
typedef ap_fixed<16, 4> accum_default_t;
typedef ap_fixed<16, 4> weight_default_t;
typedef ap_fixed<16, 4> bias_default_t;
typedef ap_fixed<16, 4> data_t;

#define N_LOOP 28
#define N_INPUTS 28
#define N_STATES 128
#define N_OUTPUTS 10

struct config0 : nn::lstm_config
{
	static const unsigned n_in = N_INPUTS;
	static const unsigned n_state = N_STATES;
	static const unsigned unroll_factor = 32; // unroll factor of cell update loop
	static const unsigned partition_factor = 64;
    static const unsigned n_out = N_OUTPUTS;
};

struct cell_act_config
{
	typedef data_t table_t;
	static const unsigned n_in = N_STATES;
	static const unsigned table_size = 4096;
	static const unsigned activation_type = nn::activ_tanh;
};

struct recurrent_act_config
{
    typedef data_t table_t;
    static const unsigned n_in = N_STATES * 3;
    static const unsigned table_size = 4096;
    static const unsigned activation_type = nn::activ_hard_sigmoid; // Keras default
    static const unsigned unroll_factor = 64;                       // for unrolling hardsigmoid
};

struct softmax_config
{
	static const unsigned n_in = N_OUTPUTS;
	static const unsigned table_size = 2048;
	typedef float table_t;
};
#endif

// typedef ap_fixed<10,2> data_t; 
// typedef ap_fixed<10,2> param_t; 
// typedef ap_fixed<11,3> Wi_x_t; 
// typedef ap_fixed<11,3> Ui_x_t; 
// typedef ap_fixed<11,3> i_x_t; 
// typedef ap_fixed<11,2> i_t; 
// typedef ap_fixed<10,2> Wf_x_t; 
// typedef ap_fixed<11,3> Uf_x_t; 
// typedef ap_fixed<10,4> f_x_t; 
// typedef ap_fixed<10,2> f_t; 
// typedef ap_fixed<11,3> Wo_x_t; 
// typedef ap_fixed<12,4> Uo_x_t; 
// typedef ap_fixed<11,4> o_x_t; 
// typedef ap_fixed<11,2> o_t; 
// typedef ap_fixed<11,3> Wc_x_t; 
// typedef ap_fixed<11,3> Uc_x_t; 
// typedef ap_fixed<11,4> c_x_t; 
// typedef ap_fixed<11,2> c_t; 
// typedef ap_fixed<13,5> c_new1_t; 
// typedef ap_fixed<10,2> c_new2_t; 
// typedef ap_fixed<13,5> c_new_t; 
// typedef ap_fixed<10,2> c_new_act_t; 
// typedef ap_fixed<10,2> h_new_t; 
// typedef ap_fixed<13,5> why_h_t; 
// typedef ap_fixed<13,5> y_t; 
