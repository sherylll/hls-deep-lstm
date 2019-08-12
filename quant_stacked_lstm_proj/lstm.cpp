// accuracy 4599/4890

#include "parameter.h"
#include "../hlslib/fic_utils/fic_packet.h"

#include "weights/lstm0/Wi.h"
#include "weights/lstm0/Wf.h"
#include "weights/lstm0/Wc.h"
#include "weights/lstm0/Wo.h"
#include "weights/lstm0/Ui.h"
#include "weights/lstm0/Uf.h"
#include "weights/lstm0/Uc.h"
#include "weights/lstm0/Uo.h"
#include "weights/lstm0/bi.h"
#include "weights/lstm0/bf.h"
#include "weights/lstm0/bc.h"
#include "weights/lstm0/bo.h"
#include "weights/lstm0/proj.h"

#include "weights/lstm1/Wi.h"
#include "weights/lstm1/Wf.h"
#include "weights/lstm1/Wc.h"
#include "weights/lstm1/Wo.h"
#include "weights/lstm1/Ui.h"
#include "weights/lstm1/Uf.h"
#include "weights/lstm1/Uc.h"
#include "weights/lstm1/Uo.h"
#include "weights/lstm1/bi.h"
#include "weights/lstm1/bf.h"
#include "weights/lstm1/bc.h"
#include "weights/lstm1/bo.h"
#include "weights/lstm1/proj.h"

#include "weights/fc/by.h"
#include "weights/fc/Wy.h"

//#define DRAM_BITWIDTH N_INPUTS * 8
#define DRAM_BITWIDTH 256

ap_uint<4> max_likelihood(prob_t y[N_OUTPUTS])
{
	ap_uint<4> i_likely = 0;
	prob_t y_max = 0;
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

void lstmp(data_t data[N_LOOP][N_INPUTS],
		ap_uint<4> *res)
{
#pragma HLS INTERFACE axis port = data
#pragma HLS INTERFACE axis port = res

#pragma HLS ARRAY_PARTITION variable=Ui,Uf,Wc,Wo,Wi,Wf,Wc,Wo complete dim = 2
#pragma HLS ARRAY_PARTITION variable=bi,bf,bc,bo complete

   static lstm_t h0_oldstate[N_PROJ] = {0};
   static lstm_t c0_oldstate[N_STATES] = {0};
    lstm_t h0_newstate[N_PROJ] = {0};
    lstm_t c0_newstate[N_STATES] = {0};

   static lstm_t h1_oldstate[N_PROJ] = {0};
   static lstm_t c1_oldstate[N_STATES] = {0};
    lstm_t h1_newstate[N_PROJ] = {0};
    lstm_t c1_newstate[N_STATES] = {0};

#pragma HLS ARRAY_PARTITION variable = h0_oldstate,c0_oldstate,h0_newstate,c0_newstate complete
#pragma HLS ARRAY_PARTITION variable = h1_oldstate,c1_oldstate,h1_newstate,c1_newstate complete

   ap_uint<169> h0_packets[N_PACKETS];
   static short timestep = 0;
   for (int i = 0; i<N_LOOP; i++)
   {
		nn::lstmp<data_t, lstm_t, config0, cell_act_config, recurrent_act_config>(data[i], h0_oldstate, h0_newstate, c0_oldstate, c0_newstate,
				Wi, Wf, Wc, Wo, Ui, Uf, Uc, Uo, proj, bi, bf, bc, bo);
	    fic::encoder<lstm_t, packet_config>(h0_newstate, h0_packets);
		fic::decoder<lstm_t, packet_config>(h0_packets, h0_newstate);

       nn::lstmp<lstm_t, lstm_t, config1, cell_act_config, recurrent_act_config>(h0_newstate, h1_oldstate, h1_newstate, c1_oldstate, c1_newstate,
				Wi1, Wf1, Wc1, Wo1, Ui1, Uf1, Uc1, Uo1, proj1, bi1, bf1, bc1, bo1);
   }
   res_t y[N_OUTPUTS] = {0};
   prob_t prob[N_OUTPUTS];
   nn::fc<lstm_t, res_t, fc_config0>(Wy, h1_newstate, by, y);
//    for(int i=0; i<N_STATES; i++) std::cout << h1_newstate[i].range() << " "; std::cout<<std::endl;
    nn::softmax<res_t, prob_t, softmax_config>(y, prob);
    for(int i=0; i<N_OUTPUTS; i++) std::cout << prob[i] << " "; std::cout<<std::endl;
    for(int i=0; i<N_OUTPUTS; i++) std::cout << prob[i].range() << " "; std::cout<<std::endl;

   * res = max_likelihood(prob);

//    nn::softmax<data_t, data_t, softmax_config>(y,res);
   for (int i = 0; i < N_PROJ; i++)
   {
#pragma HLS unroll
       h0_oldstate[i] = 0; h0_newstate[i] = 0;
       h1_oldstate[i] = 0; h1_newstate[i] = 0;
   }
   for (int i = 0; i < N_STATES; i++)
   {
#pragma HLS unroll
       c0_oldstate[i] = 0; c0_newstate[i] = 0;
       c1_oldstate[i] = 0; c1_newstate[i] = 0;
   }
}
// void lstmp(data_t data[N_LOOP][N_INPUTS],
// 		ap_uint<4> *res){}
