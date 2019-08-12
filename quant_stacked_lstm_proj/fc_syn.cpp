#include "parameter.h"
#include "weights/fc/by.h"
#include "weights/fc/Wy.h"
#include "../hlslib/fic_utils/fic_packet.h"

//void max_likelihood(res_t y[N_OUTPUTS], ap_uint<4> res[1])
//{
//	ap_uint<4> i_likely = 0;
//	res_t y_max = -16;
//	for (int i = 0; i < N_OUTPUTS; i++)
//	{
//#pragma HLS unroll
//		y_max = (y[i] > y_max) ? y[i] : y_max;
//	}
//	for (int i = 0; i < N_OUTPUTS; i++){
//		if (y_max == y[i])
//		{ i_likely = i; break; }
//	}
//	res[0] = i_likely;
//}

void fc(ap_uint<169> h1_packets[N_PACKETS],
		prob_t prob[N_OUTPUTS])
{
#pragma HLS INTERFACE axis port = h1_packets
#pragma HLS INTERFACE axis port = prob

#pragma HLS ARRAY_PARTITION variable=Wy complete dim = 2
#pragma HLS ARRAY_PARTITION variable=by complete
    lstm_t h1_states[N_PROJ];
#pragma HLS ARRAY_PARTITION variable =h1_states complete

    res_t y[N_OUTPUTS] = {0};
#pragma HLS ARRAY_PARTITION variable =y complete

	fic::decoder<lstm_t, packet_config>(h1_packets, h1_states);
    nn::fc<lstm_t, res_t, fc_config0>(Wy, h1_states, by, y);
    nn::softmax<res_t, prob_t, softmax_config>(y, prob); // 0.99 is represented, for e.g.,as 127. 
    // when sending 8-bit numbers to 4-bit RPi GPIO results in half of the bits getting discarded... 
    // sending the upper 4-bits only. Shoule be enough for our purposes
}
