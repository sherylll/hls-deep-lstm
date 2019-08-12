#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include "ap_fixed.h"
//#include "ap_cint.h"

typedef ap_fixed<8, 4, AP_RND_CONV, AP_SAT>  lstm_t; // for lstm
typedef ap_fixed<8, 1, AP_RND_CONV, AP_SAT>  bias_t;
typedef ap_fixed<8, 1, AP_RND_CONV, AP_SAT>  kernel_t;
typedef ap_fixed<8, 1, AP_RND_CONV, AP_SAT>  by_t;
typedef ap_fixed<8, 2, AP_RND_CONV, AP_SAT>  Wy_t;
typedef ap_fixed<32, 6, AP_RND_CONV, AP_SAT>  data_t;
typedef ap_fixed<8, 6, AP_RND_CONV, AP_SAT>  res_t;

#include "../hlslib/nn_utils/nn_common.h"
#include "../hlslib/nn_utils/nn_recurrent.h"

#define N_LOOP 98
#define N_INPUTS 16 // testing synthesis
#define N_STATES 128
#define N_OUTPUTS 12


//#define N_READCOUNTS 2
//
//struct dram_config
//{
//	static const unsigned n_data = N_INPUTS;
//	static const unsigned n_readcount = N_READCOUNTS;
//	static const unsigned n_chunks = 128 / DATA_TYPE_SIZE;
//	static const unsigned len_data = DATA_TYPE_SIZE;
//};

struct config0 : nn::lstm_config
{
	typedef kernel_t kernel_T;
	typedef bias_t bias_T;
	static const unsigned n_in = N_INPUTS;
	static const unsigned n_state = N_STATES;
//	static const unsigned unroll_factor = 32; // unroll factor of cell update loop
	static const unsigned partition_factor = 64;
    static const unsigned n_out = N_OUTPUTS;
};

struct fc_config0 : nn::fc_config
{
    typedef by_t bias_t;
    typedef Wy_t weight_t;
    typedef ap_fixed<16,8, AP_RND_CONV, AP_SAT> accum_t;
	static const unsigned n_in = N_STATES;
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
    static const unsigned activation_type = nn::activ_sigmoid;
};

#endif
