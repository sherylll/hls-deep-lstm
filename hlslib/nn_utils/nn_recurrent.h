// author: Yuxi Sun
// Based on nnet_utils from HLS4ML

#ifndef NNET_RECURRENT_H_
#define NNET_RECURRENT_H_

#include "nn_common.h"
#include "nn_activation.h"
namespace nn
{

struct rnn_config
{
    // Layer Sizes
    static const unsigned n_in = 10;
    static const unsigned n_state = 2;
    static const unsigned activation_type = nn::activ_tanh;

    // pipeline init interval
    static const unsigned ii_factor = 1;
};

struct tanh_act_config
{
        typedef ap_fixed<16,6> table_t;
        static const unsigned n_in = 64;
        static const unsigned table_size = 2048;
        static const unsigned activation_type = nn::activ_tanh;
};


// raw cell that returns new states
template <class data_T, typename CONFIG_T>
void vanilla_rnn(data_T Wxh[CONFIG_T::n_state][CONFIG_T::n_in],
                 data_T Whh[CONFIG_T::n_state][CONFIG_T::n_state],
                 data_T bh[CONFIG_T::n_state],
                 data_T h_curr[CONFIG_T::n_state],
                 data_T h_last[CONFIG_T::n_state],
                 data_T data[CONFIG_T::n_in])
{
//#pragma HLS inline off // why?

    data_T Wxh_dot_x_sum, Whh_dot_h_sum;
    data_T Wxh_dot_x[CONFIG_T::n_in], Whh_dot_h[CONFIG_T::n_state];
#pragma HLS ARRAY_PARTITION variable = Wxh_dot_x complete dim = 1
#pragma HLS ARRAY_PARTITION variable = Whh_dot_h cyclic factor = 10 dim = 1
    data_T h_tmp_in[CONFIG_T::n_state];
#pragma HLS ARRAY_PARTITION variable = h_tmp_in complete
forward_label0:
    for (int i = 0; i < CONFIG_T::n_state; i++)
    {
        // does it work with small intervals? why II=1 consumes so much DSP?
#pragma HLS PIPELINE II = CONFIG_T::ii_factor
        Wxh_dot_x_sum = 0;
        Whh_dot_h_sum = 0;
        for (int j = 0; j < CONFIG_T::n_in; j++)
        {
            Wxh_dot_x[j] = Wxh[i][j] * data[j];
        }

        for (int j = 0; j < CONFIG_T::n_state; j++)
        {
            Whh_dot_h[j] = Whh[i][j] * h_last[j];
        }

        step10_sum<data_T>(Wxh_dot_x, &Wxh_dot_x_sum, CONFIG_T::n_in);

        step10_sum<data_T>(Whh_dot_h, &Whh_dot_h_sum, CONFIG_T::n_state);

        h_tmp_in[i] = Wxh_dot_x_sum + Whh_dot_h_sum + bh[i];
//        h_curr[i] = hls::tanh((float)h_tmp_in);
    }
    tanh<data_T, data_T, tanh_act_config>(h_tmp_in, h_curr);
    for (int i = 0; i < CONFIG_T::n_state; i++)
    {
#pragma HLS unroll
        h_last[i] = h_curr[i];
    }
}

struct lstm_config
{
    // Internal data type definitions
    typedef float weight_t;
    typedef float bias_t;

    // Layer Sizes
    static const unsigned n_in = 2;
    static const unsigned n_out = 2;
    static const unsigned n_state = 2;
    static const unsigned n_4state = 8;
    static const unsigned table_size = 1024;
    static const unsigned partition_factor=8;
};

template <class data_T, typename CONFIG_T, typename ACT_CONFIG_C, typename ACT_CONFIG_IFO>
void lstm_static(data_T data[CONFIG_T::n_in],
                 data_T h_oldstate[CONFIG_T::n_state],
				 data_T h_newstate[CONFIG_T::n_state], // to be sent over the switch
                 data_T s_oldstate[CONFIG_T::n_state],
				 data_T s_newstate[CONFIG_T::n_state],
                 data_T param_i[CONFIG_T::n_state][CONFIG_T::n_in],/*W ifco*/
				 data_T param_f[CONFIG_T::n_state][CONFIG_T::n_in],
				 data_T param_c[CONFIG_T::n_state][CONFIG_T::n_in],
				 data_T param_o[CONFIG_T::n_state][CONFIG_T::n_in],

                 data_T param_r_i[CONFIG_T::n_state][CONFIG_T::n_state], /*U ifco*/
				 data_T param_r_f[CONFIG_T::n_state][CONFIG_T::n_state],
				 data_T param_r_c[CONFIG_T::n_state][CONFIG_T::n_state],
				 data_T param_r_o[CONFIG_T::n_state][CONFIG_T::n_state],

				 data_T param_b_i[CONFIG_T::n_state],/*bias ifco*/
				 data_T param_b_f[CONFIG_T::n_state],
				 data_T param_b_c[CONFIG_T::n_state],
				 data_T param_b_o[CONFIG_T::n_state])
{
//#pragma HLS function_instantiate variable=param_i,param_f,param_c,param_o,param_r_i,param_r_f,param_r_c,param_r_o,param_b_i,param_b_f,param_b_c,param_b_o
	data_T fc_i[CONFIG_T::n_state], fc_f[CONFIG_T::n_state], fc_c[CONFIG_T::n_state], fc_o[CONFIG_T::n_state];
	data_T fc_i_state[CONFIG_T::n_state], fc_f_state[CONFIG_T::n_state], fc_c_state[CONFIG_T::n_state], fc_o_state[CONFIG_T::n_state];
#pragma HLS ARRAY_PARTITION variable = fc_i,fc_f,fc_c,fc_o cyclic factor=CONFIG_T::partition_factor
#pragma HLS ARRAY_PARTITION variable = fc_i_state,fc_f_state,fc_c_state,fc_o_state cyclic factor=CONFIG_T::partition_factor

    data_T tmpres_ifo[CONFIG_T::n_state * 3];   // activated i,f,o matrices (input, forget output)
    data_T tmpres_c[CONFIG_T::n_state];         // activated c-matrix (keras notation)
    data_T inputacc_ifo[CONFIG_T::n_state * 3]; // i,f,o matrices (keras notation)
    data_T inputacc_c[CONFIG_T::n_state];       // c-matrix (keras notation)
    data_T s_actstate[CONFIG_T::n_state];

#pragma HLS ARRAY_PARTITION variable = tmpres_ifo cyclic factor=CONFIG_T::partition_factor
#pragma HLS ARRAY_PARTITION variable = tmpres_c cyclic factor=CONFIG_T::partition_factor
#pragma HLS ARRAY_PARTITION variable = inputacc_ifo cyclic factor=CONFIG_T::partition_factor
#pragma HLS ARRAY_PARTITION variable = inputacc_c cyclic factor=CONFIG_T::partition_factor
#pragma HLS ARRAY_PARTITION variable = s_actstate cyclic factor=CONFIG_T::partition_factor

    // reduce access to argument on interface
    data_T data_temp[CONFIG_T::n_in];
#pragma HLS ARRAY_PARTITION variable = data_temp complete
    for (int i=0; i<CONFIG_T::n_in; i++)
#pragma HLS unroll
    	data_temp[i]=data[i];

    // [W_i, W_f, W_c, W_o] * x + [b_i, b_f, b_c, b_o]
    fc_no_b<data_T, CONFIG_T::n_in, CONFIG_T::n_state>(param_i, data_temp, fc_i);
    fc_no_b<data_T, CONFIG_T::n_in, CONFIG_T::n_state>(param_f, data_temp, fc_f);
    fc_no_b<data_T, CONFIG_T::n_in, CONFIG_T::n_state>(param_c, data_temp, fc_c);
    fc_no_b<data_T, CONFIG_T::n_in, CONFIG_T::n_state>(param_o, data_temp, fc_o);

    // [U_i, U_f, U_c, U_o] * h
    fc_no_b<data_T, CONFIG_T::n_state, CONFIG_T::n_state>(param_r_i, h_oldstate, fc_i_state);
    fc_no_b<data_T, CONFIG_T::n_state, CONFIG_T::n_state>(param_r_f, h_oldstate, fc_f_state);
    fc_no_b<data_T, CONFIG_T::n_state, CONFIG_T::n_state>(param_r_c, h_oldstate, fc_c_state);
    fc_no_b<data_T, CONFIG_T::n_state, CONFIG_T::n_state>(param_r_o, h_oldstate, fc_o_state);

    // [W_i, W_f, W_c, W_o] * x + [U_i, U_f,U_c, U_o] * x + [b_i, b_f, b_c, b_o]
    for (int i=0; i<CONFIG_T::n_state; i++)
    {
#pragma HLS unroll
    	inputacc_ifo[i] = fc_i[i]+ fc_i_state[i] + param_b_i[i];
    	inputacc_ifo[i+CONFIG_T::n_state] = fc_f[i]+ fc_f_state[i] + param_b_f[i];
    	inputacc_ifo[i+CONFIG_T::n_state*2] = fc_o[i]+ fc_o_state[i] + param_b_o[i];
    	inputacc_c[i] =fc_c[i] + fc_c_state[i] + param_b_c[i];
    }

    // recurrent_activation (keras defaults to hard sigmoid)
    if (ACT_CONFIG_IFO::activation_type == activ_hard_sigmoid)
        hard_sigmoid<data_T, data_T, ACT_CONFIG_IFO>(inputacc_ifo, tmpres_ifo);
    // TODO test more activation types

    // activation
    if (ACT_CONFIG_C::activation_type == activ_tanh)
    {
        // TODO change to table after quant.
//        for (int i = 0; i < CONFIG_T::n_state; i++)
//#pragma HLS pipeline
//            tmpres_c[i] = hls::tanh((float)inputacc_c[i]);
    	tanh<data_T, data_T, ACT_CONFIG_C>(inputacc_c, tmpres_c);
    }

    // c = i .* act(W_c * x + U_c * h_old + b_c) + f .* c_old
CELL_UPDATE_LOOP:for (int iacc = 0; iacc < (CONFIG_T::n_state); iacc++)
    {
#pragma HLS UNROLL factor=CONFIG_T::unroll_factor
	   data_T temp1 = tmpres_c[iacc] * tmpres_ifo[iacc];
	   data_T temp2 = s_oldstate[iacc] * tmpres_ifo[iacc + (CONFIG_T::n_state)];
	   data_T temp_s = temp1+temp2;
#pragma HLS RESOURCE variable=temp_s core=AddSub
        s_newstate[iacc] = temp_s;
    }

    // h = act(c) * o
    if (ACT_CONFIG_C::activation_type == activ_tanh)
    {
        // TODO change to table after quant.
//        for (int i = 0; i < CONFIG_T::n_state; i++)
//#pragma HLS pipeline
//            s_actstate[i] = hls::tanh((float)s_newstate[i]);
    	tanh<data_T, data_T, ACT_CONFIG_C>(s_newstate, s_actstate);
    }

    for (int iacc = 0; iacc < CONFIG_T::n_state; iacc++)
    {
#pragma HLS UNROLL factor=CONFIG_T::unroll_factor
        h_newstate[iacc] = tmpres_ifo[iacc + 2 * (CONFIG_T::n_state)] * s_actstate[iacc];
    }

    for (int i = 0; i < CONFIG_T::n_state; i++)
    {
#pragma HLS unroll
    	h_oldstate[i] = h_newstate[i];
    	s_oldstate[i] = s_newstate[i];
}

//            std::cout << "Post-State: s [ "; for (int ii = 0; ii < CONFIG_T::n_state; ii++) std::cout << s_newstate[ii] << " "; std::cout << "]" << std::endl;
//            std::cout << "Post-State: h [ "; for (int ii = 0; ii < CONFIG_T::n_state; ii++) std::cout << h_newstate[ii] << " "; std::cout << "]" << std::endl;
}
} // namespace nn

#endif
