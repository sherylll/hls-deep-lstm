#include "parameter.h"
#include "../hlslib/fic_utils/fic_packet.h"

#include "weights/Wi.h"
#include "weights/Wf.h"
#include "weights/Wc.h"
#include "weights/Wo.h"
#include "weights/Ui.h"
#include "weights/Uf.h"
#include "weights/Uc.h"
#include "weights/Uo.h"
#include "weights/bi.h"
#include "weights/bf.h"
#include "weights/bc.h"
#include "weights/bo.h"

#include "weights/by.h"
#include "weights/Wy.h"

void lstm0(data_t data[N_LOOP][N_INPUTS],
           data_t res[N_OUTPUTS])
{
#pragma HLS INTERFACE axis port = data
#pragma HLS INTERFACE axis port = res

 #pragma HLS ARRAY_PARTITION variable = W_i complete dim = 2
 #pragma HLS ARRAY_PARTITION variable = W_f complete dim = 2
 #pragma HLS ARRAY_PARTITION variable = W_c complete dim = 2
 #pragma HLS ARRAY_PARTITION variable = W_o complete dim = 2

 #pragma HLS ARRAY_PARTITION variable = U_i cyclic factor=64 dim = 2
 #pragma HLS ARRAY_PARTITION variable = U_f cyclic factor=64 dim = 2
 #pragma HLS ARRAY_PARTITION variable = U_c cyclic factor=64 dim = 2
 #pragma HLS ARRAY_PARTITION variable = U_o cyclic factor=64 dim = 2

 #pragma HLS ARRAY_PARTITION variable = b_i cyclic factor=64 dim = 1
 #pragma HLS ARRAY_PARTITION variable = b_f cyclic factor=64 dim = 1
 #pragma HLS ARRAY_PARTITION variable = b_c cyclic factor=64 dim = 1
 #pragma HLS ARRAY_PARTITION variable = b_o cyclic factor=64 dim = 1

    static lstm_t h0_oldstate[N_STATES] = {0};
    static lstm_t c0_oldstate[N_STATES] = {0};

    static lstm_t h0_newstate[N_STATES] = {0};
    static lstm_t c0_newstate[N_STATES] = {0};

#pragma HLS ARRAY_PARTITION variable = h0_oldstate cyclic factor=64
#pragma HLS ARRAY_PARTITION variable = c0_oldstate cyclic factor=64
#pragma HLS ARRAY_PARTITION variable = h0_newstate cyclic factor=64
#pragma HLS ARRAY_PARTITION variable = c0_newstate cyclic factor=64

    static short timestep = 0;
    for (int i = 0; i<N_LOOP; i++){
        nn::lstm_static<data_t, lstm_t, config0, cell_act_config, recurrent_act_config>(data[i], h0_oldstate, h0_newstate, c0_oldstate, c0_newstate, W_i,W_f,W_c,W_o,
        		U_i,U_f,U_c,U_o, b_i, b_f, b_c, b_o);
    }
    data_t y[N_OUTPUTS] = {0};
//    nn::fc<lstm_t, Wy_t, by_t, data_t, config0::n_state, config0::n_out>(W_y, h0_newstate, b_y, res);
    nn::fc<lstm_t, data_t, fc_config0>(W_y, h0_newstate, b_y, res);

//    nn::softmax<data_t, data_t, softmax_config>(y,res);
    for (int i = 0; i < N_STATES; i++)
    {
#pragma HLS unroll
        h0_oldstate[i] = 0;
        c0_oldstate[i] = 0;
        h0_newstate[i] = 0;
        c0_newstate[i] = 0;
    }
}
