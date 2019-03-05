#include "parameters.h"
#include "../../hlslib/fic_utils/fic_packet.h"

void lstm0(data_t data[N_LOOP][N_INPUTS],
           data_t res[N_OUTPUTS])
{
#pragma HLS INTERFACE axis port = data
#pragma HLS INTERFACE axis port = res

    data_t W_i[N_STATES][N_INPUTS]=W_I, W_f[N_STATES][N_INPUTS]=W_F, W_c[N_STATES][N_INPUTS]=W_C, W_o[N_STATES][N_INPUTS]=W_O;
    data_t U_i[N_STATES][N_STATES]=U_I, U_f[N_STATES][N_STATES]=U_F, U_c[N_STATES][N_STATES]=U_C, U_o[N_STATES][N_STATES]=U_O;
    data_t b_i[N_STATES]=B_I, b_f[N_STATES]=B_F, b_c[N_STATES]=B_C, b_o[N_STATES]=B_O;

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

    static data_t h0_oldstate[N_STATES] = {0};
    static data_t c0_oldstate[N_STATES] = {0};

    static data_t h0_newstate[N_STATES] = {0};
    static data_t c0_newstate[N_STATES] = {0};

#pragma HLS ARRAY_PARTITION variable = h0_oldstate cyclic factor=64
#pragma HLS ARRAY_PARTITION variable = c0_oldstate cyclic factor=64
#pragma HLS ARRAY_PARTITION variable = h0_newstate cyclic factor=64
#pragma HLS ARRAY_PARTITION variable = c0_newstate cyclic factor=64

    static short timestep = 0;
    for (int i = 0; i<N_LOOP; i++){
        nn::lstm_static<data_t, config0, cell_act_config, recurrent_act_config>(data[iloop], h0_oldstate, h0_newstate, c0_oldstate, c0_newstate, W_i,W_f,W_c,W_o,
        		U_i,U_f,U_c,U_o, b_i, b_f, b_c, b_o);
    }
    data_t y[N_OUTPUTS] = {0};
    nn:fc<data_t, config0::n_state, config0::n_out>(Why, h0_newstate, by, y);
    nn::softmax<data_t, data_t, softmax_config>(y,res);
    for (int i = 0; i < N_STATES; i++)
    {
#pragma HLS unroll
        h0_oldstate[i] = 0;
        c0_oldstate[i] = 0;
        h0_newstate[i] = 0;
        c0_newstate[i] = 0;
    }
}
