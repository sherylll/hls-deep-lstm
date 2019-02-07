#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include "ap_fixed.h"
#include "../../hlslib/nn_utils/nn_common.h"
#include "../../hlslib/nn_utils/nn_recurrent.h"

#include "../../stacked_lstm_weights/cell0/W_i.h"
#include "../../stacked_lstm_weights/cell0/W_f.h"
#include "../../stacked_lstm_weights/cell0/W_c.h"
#include "../../stacked_lstm_weights/cell0/W_o.h"
#include "../../stacked_lstm_weights/cell0/U_i.h"
#include "../../stacked_lstm_weights/cell0/U_f.h"
#include "../../stacked_lstm_weights/cell0/U_c.h"
#include "../../stacked_lstm_weights/cell0/U_o.h"
#include "../../stacked_lstm_weights/cell0/b_i.h"
#include "../../stacked_lstm_weights/cell0/b_f.h"
#include "../../stacked_lstm_weights/cell0/b_c.h"
#include "../../stacked_lstm_weights/cell0/b_o.h"

#include "../../stacked_lstm_weights/cell1/W_i.h"
#include "../../stacked_lstm_weights/cell1/W_f.h"
#include "../../stacked_lstm_weights/cell1/W_c.h"
#include "../../stacked_lstm_weights/cell1/W_o.h"
#include "../../stacked_lstm_weights/cell1/U_i.h"
#include "../../stacked_lstm_weights/cell1/U_f.h"
#include "../../stacked_lstm_weights/cell1/U_c.h"
#include "../../stacked_lstm_weights/cell1/U_o.h"
#include "../../stacked_lstm_weights/cell1/b_i.h"
#include "../../stacked_lstm_weights/cell1/b_f.h"
#include "../../stacked_lstm_weights/cell1/b_c.h"
#include "../../stacked_lstm_weights/cell1/b_o.h"

#include "../../stacked_lstm_weights/cell2/W_i.h"
#include "../../stacked_lstm_weights/cell2/W_f.h"
#include "../../stacked_lstm_weights/cell2/W_c.h"
#include "../../stacked_lstm_weights/cell2/W_o.h"
#include "../../stacked_lstm_weights/cell2/U_i.h"
#include "../../stacked_lstm_weights/cell2/U_f.h"
#include "../../stacked_lstm_weights/cell2/U_c.h"
#include "../../stacked_lstm_weights/cell2/U_o.h"
#include "../../stacked_lstm_weights/cell2/b_i.h"
#include "../../stacked_lstm_weights/cell2/b_f.h"
#include "../../stacked_lstm_weights/cell2/b_c.h"
#include "../../stacked_lstm_weights/cell2/b_o.h"

#include "../../stacked_lstm_weights/cell3/W_i.h"
#include "../../stacked_lstm_weights/cell3/W_f.h"
#include "../../stacked_lstm_weights/cell3/W_c.h"
#include "../../stacked_lstm_weights/cell3/W_o.h"
#include "../../stacked_lstm_weights/cell3/U_i.h"
#include "../../stacked_lstm_weights/cell3/U_f.h"
#include "../../stacked_lstm_weights/cell3/U_c.h"
#include "../../stacked_lstm_weights/cell3/U_o.h"
#include "../../stacked_lstm_weights/cell3/b_i.h"
#include "../../stacked_lstm_weights/cell3/b_f.h"
#include "../../stacked_lstm_weights/cell3/b_c.h"
#include "../../stacked_lstm_weights/cell3/b_o.h"

#include "../../stacked_lstm_weights/fc/by.h"
#include "../../stacked_lstm_weights/fc/Why.h"

// using fixed only with caution! accuracy w/o quant is very low!
typedef ap_fixed<16, 4> accum_default_t;
typedef ap_fixed<16, 4> weight_default_t;
typedef ap_fixed<16, 4> bias_default_t;
typedef ap_fixed<16, 4> data_t;
//typedef float accum_default_t;
//typedef float weight_default_t;
//typedef float bias_default_t;
//typedef float data_t;

#define DATA_TYPE_SIZE 16
#define N_PACKETS DATA_TYPE_SIZE*N_STATES/128

#define N_LOOP 28
#define N_INPUTS 28
#define N_STATES 128
#define N_OUTPUTS 10

struct packet_config
{
	static const unsigned n_data = N_STATES;
	static const unsigned n_packets = N_PACKETS;
	static const unsigned n_chunks = 128 / DATA_TYPE_SIZE;
	static const unsigned len_data = DATA_TYPE_SIZE;
};


struct config0 : nn::lstm_config
{
	static const unsigned n_in = N_INPUTS;
	static const unsigned n_state = N_STATES;
	static const unsigned unroll_factor = 32; // unroll factor of cell update loop
	static const unsigned partition_factor = 64;
//        static const unsigned n_out = N_OUTPUTS;
};

struct config1 : nn::lstm_config
{
	static const unsigned n_in = N_STATES;
	static const unsigned n_state = N_STATES;
	static const unsigned n_out = N_OUTPUTS;
	static const unsigned unroll_factor = 32;
	static const unsigned partition_factor = 16;
};

/* *
 * the configuration given here correspond to the following Keras code:
 *
 *   inputs =  Input(shape=(img_rows, img_cols))
 *   layer = LSTM(hidden_size,  activation='tanh', unroll=True)(inputs)
 *   predictions = Dense(num_classes, activation='softmax')(layer)
 *   model = Model(inputs=inputs, outputs=predictions)
 *
 * */

struct cell_act_config
{
	typedef data_t table_t;
	static const unsigned n_in = N_STATES;
	static const unsigned table_size = 4096;
	static const unsigned activation_type = nn::activ_tanh;
};

struct softmax_config
{
	static const unsigned n_in = N_OUTPUTS;
	static const unsigned table_size = 2048;
	typedef float table_t;
};
#endif
