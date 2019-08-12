#include "parameter.h"
#include "../hlslib/fic_utils/fic_packet.h"

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

void lstm1(ap_uint<169> h0_packets[N_PACKETS],
			ap_uint<169> h1_packets[N_PACKETS])
		// lstm_t y_debug[N_PROJ])
{
#pragma HLS INTERFACE axis port = h0_packets
#pragma HLS INTERFACE axis port = h1_packets

#pragma HLS ARRAY_PARTITION variable=Ui1,Uf1,Uc1,Uo1,Wi1,Wf1,Wc1,Wo1,proj1 complete dim = 2
#pragma HLS ARRAY_PARTITION variable=bi1,bf1,bc1,bo1 cyclic factor=64

    static lstm_t h1_oldstate[N_PROJ] = {0};
    static lstm_t c1_oldstate[N_STATES] = {0};

    lstm_t h1_newstate[N_PROJ] = {0}; // creates dependency when data is encoded and sent to the switch, not very useful here...
    lstm_t c1_newstate[N_STATES] = {0};
#pragma HLS ARRAY_PARTITION variable=h1_oldstate,c1_oldstate,h1_newstate,c1_newstate complete

    lstm_t h0_states[N_PROJ];
#pragma HLS ARRAY_PARTITION variable = h0_states complete

    static unsigned short timestep = 0;

	fic::decoder<lstm_t, packet_config>(h0_packets, h0_states);
	nn::lstmp<lstm_t, lstm_t, config1, cell_act_config, recurrent_act_config>(h0_states, h1_oldstate, h1_newstate, c1_oldstate, c1_newstate,
			Wi1, Wf1, Wc1, Wo1, Ui1, Uf1, Uc1, Uo1, proj1, bi1, bf1, bc1, bo1);
	timestep += 1;
    fic::encoder<lstm_t, packet_config>(h1_newstate, h1_packets);

    if (timestep == N_LOOP) // can HLS properly build an FSM???
    {
//        for (int i = 0; i < N_PROJ; i++) // write out
//        {
// #pragma HLS unroll
//     	   y_debug[i] = h1_newstate[i];
//     	   // 246 247 241 232 28 250 20 235 242 4 245 21 8 9 22 6 15 252 6 247 13 244 32 20 11 23 251 244 252 1 5 239 11 253 38 254 2 26 226 2 250 242 254 16 6 240 218 8 18 252 255 8 232 9 248 251 255 255 12 3 252 239 253 234 223 23 0 246 30 228 25 12 236 252 1 243 0 0 0 0 0 0 0 0 0 0 0 0 48 248 111 164 252 127 0 0 192 248 111 164 252 127 0 0 0 0 0 0 7 0 0 0 0 0 112 164 252 127 0 0 0 247 111 164 252 127 0 0
//        }
	   for (int i = 0; i < N_PROJ; i++)
	   {
#pragma HLS unroll
		   h1_oldstate[i] = 0; h1_newstate[i] = 0;
	   }
	   for (int i = 0; i < N_STATES; i++)
	   {
#pragma HLS unroll
		   c1_oldstate[i] = 0; c1_newstate[i] = 0;
	   }
    	timestep = 0;
    }
}
