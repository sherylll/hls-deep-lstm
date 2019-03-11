#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include "ap_fixed.h"
#include "../hlslib/nn_utils/nn_common.h"
#include "../hlslib/nn_utils/nn_recurrent.h"
//#include "param_types.h"

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
 typedef ap_fixed<10, 5, AP_RND_CONV, AP_SAT> data_t;

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

//template <typename CONFIG_T, typename ACT_CONFIG_C, typename ACT_CONFIG_IFO>
//void lstm_quant(data_t data[CONFIG_T::n_in],
//                h_new_t h_oldstate[CONFIG_T::n_state],
//				 h_new_t h_newstate[CONFIG_T::n_state], // to be sent over the switch
//                c_new_act_t s_oldstate[CONFIG_T::n_state],
//				 c_new_act_t s_newstate[CONFIG_T::n_state],
//
//				 param_t w_i[CONFIG_T::n_state][CONFIG_T::n_in],/*W ifco*/
//				 param_t w_f[CONFIG_T::n_state][CONFIG_T::n_in],
//				 param_t w_c[CONFIG_T::n_state][CONFIG_T::n_in],
//				 param_t w_o[CONFIG_T::n_state][CONFIG_T::n_in],
//
//				 param_t u_i[CONFIG_T::n_state][CONFIG_T::n_state], /*U ifco*/
//				 param_t u_f[CONFIG_T::n_state][CONFIG_T::n_state],
//				 param_t u_c[CONFIG_T::n_state][CONFIG_T::n_state],
//				 param_t u_o[CONFIG_T::n_state][CONFIG_T::n_state],
//
//				 param_t b_i[CONFIG_T::n_state],/*bias ifco*/
//				 param_t b_f[CONFIG_T::n_state],
//				 param_t b_c[CONFIG_T::n_state],
//				 param_t b_o[CONFIG_T::n_state])
//{
//	Wi_x_t fc_i[CONFIG_T::n_state]; Wf_x_t fc_f[CONFIG_T::n_state]; Wc_x_t fc_c[CONFIG_T::n_state]; Wo_x_t fc_o[CONFIG_T::n_state];
//	Ui_x_t fc_i_state[CONFIG_T::n_state]; Uf_x_t[CONFIG_T::n_state]; Uc_x_t fc_c_state[CONFIG_T::n_state]; Uo_x_t fc_o_state[CONFIG_T::n_state];
//
//#pragma HLS ARRAY_PARTITION variable = fc_i,fc_f,fc_c,fc_o cyclic factor=CONFIG_T::partition_factor
//#pragma HLS ARRAY_PARTITION variable = fc_i_state,fc_f_state,fc_c_state,fc_o_state cyclic factor=CONFIG_T::partition_factor
//
//   // data_T tmpres_ifo[CONFIG_T::n_state * 3];   // activated i,f,o matrices (input, forget output)
//   i_t i_gate[CONFIG_T::n_state]; f_t f_gate[CONFIG_T::n_state]; c_t c_gate[CONFIG_T::n_state]; o_t i_gate[CONFIG_T::n_state];
//
//   data_T tmpres_c[CONFIG_T::n_state];         // activated c-matrix (keras notation)
//   // data_T inputacc_ifo[CONFIG_T::n_state * 3]; // i,f,o matrices (keras notation)
//   i_x_t i_in[CONFIG_T::n_state]; f_x_t f_in[CONFIG_T::n_state]; c_x_t c_in[CONFIG_T::n_state]; o_x_t o_in[CONFIG_T::n_state];
//
//   data_T inputacc_c[CONFIG_T::n_state];       // c-matrix (keras notation)
//   data_T s_actstate[CONFIG_T::n_state];
//
//#pragma HLS ARRAY_PARTITION variable = tmpres_ifo cyclic factor=CONFIG_T::partition_factor
//#pragma HLS ARRAY_PARTITION variable = tmpres_c cyclic factor=CONFIG_T::partition_factor
//#pragma HLS ARRAY_PARTITION variable = inputacc_ifo cyclic factor=CONFIG_T::partition_factor
//#pragma HLS ARRAY_PARTITION variable = inputacc_c cyclic factor=CONFIG_T::partition_factor
//#pragma HLS ARRAY_PARTITION variable = s_actstate cyclic factor=CONFIG_T::partition_factor
//
//   // reduce access to argument on interface
//   data_t data_temp[CONFIG_T::n_in];
//#pragma HLS ARRAY_PARTITION variable = data_temp complete
//   for (int i=0; i<CONFIG_T::n_in; i++)
//#pragma HLS unroll
//   	data_temp[i]=data[i];
//
//   // [W_i, W_f, W_c, W_o] * x + [b_i, b_f, b_c, b_o]
//   fc_no_b<data_t, Wi_x_t, CONFIG_T::n_in, CONFIG_T::n_state>(w_i, data_temp, fc_i);
//   fc_no_b<data_t,Wf_x_t, CONFIG_T::n_in, CONFIG_T::n_state>(w_f, data_temp, fc_f);
//   fc_no_b<data_t,Wc_x_t, CONFIG_T::n_in, CONFIG_T::n_state>(w_c, data_temp, fc_c);
//   fc_no_b<data_t,Wo_x_t, CONFIG_T::n_in, CONFIG_T::n_state>(w_o, data_temp, fc_o);
//
//   // [U_i, U_f, U_c, U_o] * h
//   fc_no_b<data_t,Ui_x_t, CONFIG_T::n_state, CONFIG_T::n_state>(u_i, h_oldstate, fc_i_state);
//   fc_no_b<data_t,Uf_x_t, CONFIG_T::n_state, CONFIG_T::n_state>(u_f, h_oldstate, fc_f_state);
//   fc_no_b<data_t,Uc_x_t, CONFIG_T::n_state, CONFIG_T::n_state>(u_c, h_oldstate, fc_c_state);
//   fc_no_b<data_t,Uio_x_t, CONFIG_T::n_state, CONFIG_T::n_state>(u_o, h_oldstate, fc_o_state);
//
//   // [W_i, W_f, W_c, W_o] * x + [U_i, U_f,U_c, U_o] * x + [b_i, b_f, b_c, b_o]
//   for (int i=0; i<CONFIG_T::n_state; i++)
//   {
//#pragma HLS unroll
//   	i_in[i] = fc_i[i]+ fc_i_state[i] + b_i[i];
//   	f_in[i] = fc_f[i]+ fc_f_state[i] + b_f[i];
//   	o_in[i] = fc_o[i]+ fc_o_state[i] + b_o[i];
//   	c_in[i] =fc_c[i] + fc_c_state[i] + b_c[i];
//   }
//
//   // recurrent_activation (keras defaults to hard sigmoid)
//   if (ACT_CONFIG_IFO::activation_type == activ_hard_sigmoid){
//       hard_sigmoid<i_x_t, i_t, ACT_CONFIG_IFO>(i_in, i_gate);
//       hard_sigmoid<f_x_t, f_t, ACT_CONFIG_IFO>(f_in, f_gate);
//       hard_sigmoid<o_x_t, o_t, ACT_CONFIG_IFO>(o_in, o_gate);
//   }
//   // TODO test more activation types
//
//   // activation
//	tanh<c_x_t,c_t, ACT_CONFIG_C>(c_in, c_gate);
//
//   // c = i .* act(W_c * x + U_c * h_old + b_c) + f .* c_old
//CELL_UPDATE_LOOP:for (int iacc = 0; iacc < (CONFIG_T::n_state); iacc++)
//   {
//#pragma HLS UNROLL factor=CONFIG_T::unroll_factor
//	   data_T temp1 = tmpres_c[iacc] * tmpres_ifo[iacc];
//	   data_T temp2 = s_oldstate[iacc] * tmpres_ifo[iacc + (CONFIG_T::n_state)];
//	   data_T temp_s = temp1+temp2;
//#pragma HLS RESOURCE variable=temp_s core=AddSub
//       s_newstate[iacc] = temp_s;
//   }
//
//   // h = act(c) * o
//	tanh<data_T, data_T, ACT_CONFIG_C>(s_newstate, s_actstate);
//
//   for (int iacc = 0; iacc < CONFIG_T::n_state; iacc++)
//   {
//#pragma HLS UNROLL factor=CONFIG_T::unroll_factor
//       h_newstate[iacc] = tmpres_ifo[iacc + 2 * (CONFIG_T::n_state)] * s_actstate[iacc];
//   }
//
//   for (int i = 0; i < CONFIG_T::n_state; i++)
//   {
//#pragma HLS unroll
//   	h_oldstate[i] = h_newstate[i];
//   	s_oldstate[i] = s_newstate[i];
//}
//
//           std::cout << "Post-State: s [ "; for (int ii = 0; ii < CONFIG_T::n_state; ii++) std::cout << s_newstate[ii] << " "; std::cout << "]" << std::endl;
//           std::cout << "Post-State: h [ "; for (int ii = 0; ii < CONFIG_T::n_state; ii++) std::cout << h_newstate[ii] << " "; std::cout << "]" << std::endl;
//}

#endif
