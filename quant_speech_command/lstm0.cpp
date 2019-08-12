#include "parameter.h"
#include "../hlslib/fic_utils/fic_packet.h"

#include "weights/basic_lstm_cell/Wi.h"
#include "weights/basic_lstm_cell/Wf.h"
#include "weights/basic_lstm_cell/Wc.h"
#include "weights/basic_lstm_cell/Wo.h"
#include "weights/basic_lstm_cell/Ui.h"
#include "weights/basic_lstm_cell/Uf.h"
#include "weights/basic_lstm_cell/Uc.h"
#include "weights/basic_lstm_cell/Uo.h"
#include "weights/basic_lstm_cell/bi.h"
#include "weights/basic_lstm_cell/bf.h"
#include "weights/basic_lstm_cell/bc.h"
#include "weights/basic_lstm_cell/bo.h"

#include "weights/fc/by.h"
#include "weights/fc/Wy.h"

//#define DRAM_BITWIDTH N_INPUTS * 8
#define DRAM_BITWIDTH 256

ap_uint<4> max_likelihood(res_t y[N_OUTPUTS])
{
	ap_uint<4> i_likely = 0;
	data_t y_max = 0;
	for (int i = 0; i < N_OUTPUTS; i++)
	{
#pragma HLS unroll
		if (y[i] > y_max)
		{
			y_max = y[i];
			i_likely = i;
		}
	}
	return i_likely;
}

//void dram_decoder(data_t data [N_LOOP][N_INPUTS], ap_uint<DRAM_BITWIDTH> data_buf[N_LOOP])
//{
//	ap_uint<224> tmp;
//	int ij = 0;
//	for (int d=0; d<N_LOOP; d++)
//	{
//#pragma HLS pipeline
//#pragma HLS unroll factor=2
//		tmp = data_buf[d](255,32);
//		for (int j=0; j<N_INPUTS; j++) // 128/8 = 16
//		{
//			data[d][j].range() = tmp((j+1)*8-1,j*8);
//		}
//	}
//}

//void lstm0(volatile ap_uint<DRAM_BITWIDTH> *ddr,
void lstm0(data_t data[N_LOOP][N_INPUTS],
		ap_uint<4> *res) // index, goes to RPi
//           data_t res[N_OUTPUTS])
{
#pragma HLS INTERFACE axis port = data

// possible to set to custom length? burst_length == dram bitwidth?
//#pragma HLS INTERFACE m_axi port=ddr offset=direct max_write_burst_length=256 //offset=direct: The DDR address are supplied by user from outside the HLS module
#pragma HLS INTERFACE axis port = res

#pragma HLS ARRAY_PARTITION variable=Ui,Uf,Wc,Wo,Wi,Wf,Wc,Wo complete dim = 2
#pragma HLS ARRAY_PARTITION variable=bi,bf,bc,bo complete

    static lstm_t h0_oldstate[N_STATES] = {0};
    static lstm_t c0_oldstate[N_STATES] = {0};

    static lstm_t h0_newstate[N_STATES] = {0};
    static lstm_t c0_newstate[N_STATES] = {0};

#pragma HLS ARRAY_PARTITION variable = h0_oldstate complete
#pragma HLS ARRAY_PARTITION variable = c0_oldstate complete
#pragma HLS ARRAY_PARTITION variable = h0_newstate complete
#pragma HLS ARRAY_PARTITION variable = c0_newstate complete

    static short timestep = 0;
//    data_t data [N_LOOP][N_INPUTS];
//#pragma HLS ARRAY_PARTITION variable = data complete dim=2
//
//    ap_uint<DRAM_BITWIDTH> data_buf[N_LOOP];
//#pragma HLS ARRAY_PARTITION variable = data_buf complete
//
//    // TODO check if sizeof correctly returns 8 bits
//	memcpy(data_buf, (ap_uint<DRAM_BITWIDTH>*)ddr, N_LOOP*N_INPUTS*sizeof(data_t)); // assume the rest 4 bytes filled with 0
//	dram_decoder(data, data_buf);

    for (int i = 0; i<N_LOOP; i++)
    {
		nn::lstm_static<data_t, lstm_t, config0, cell_act_config, recurrent_act_config>(data[i], h0_oldstate, h0_newstate, c0_oldstate, c0_newstate,
				Wi,Wf,Wc,Wo,Ui,Uf,Uc,Uo, bi, bf, bc, bo);
    }
    res_t y[N_OUTPUTS] = {0};
#pragma HLS ARRAY_PARTITION variable = y complete
    nn::fc<lstm_t, res_t, fc_config0>(Wy, h0_newstate, by, y);
    * res = max_likelihood(y);

//    nn::softmax<data_t, data_t, softmax_config>(y,res);
    for (int i = 0; i < N_STATES; i++)
    {
#pragma HLS unroll
        h0_oldstate[i] = 0; c0_oldstate[i] = 0; h0_newstate[i] = 0; c0_newstate[i] = 0;
    }
}
