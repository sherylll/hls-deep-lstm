#include "parameters.h"
#include "lstm.h"
#include "../../hlslib/fic_utils/fic_packet.h"

void lstm1(ap_uint<169> h0_in[N_PACKETS],
            ap_uint<169> h1_out[N_PACKETS])
            // data_t h1_out[N_STATES])
{
#pragma HLS INTERFACE axis port = h0_in
#pragma HLS INTERFACE axis port = h1_out
#pragma HLS dataflow

    data_t W1_i[N_STATES][N_STATES]=W1_I, W1_f[N_STATES][N_STATES]=W1_F, W1_c[N_STATES][N_STATES]=W1_C, W1_o[N_STATES][N_STATES]=W1_O;
    data_t U1_i[N_STATES][N_STATES]=U1_I, U1_f[N_STATES][N_STATES]=U1_F, U1_c[N_STATES][N_STATES]=U1_C, U1_o[N_STATES][N_STATES]=U1_O;
    data_t b1_i[N_STATES]=B1_I, b1_f[N_STATES]=B1_F, b1_c[N_STATES]=B1_C, b1_o[N_STATES]=B1_O;

 #pragma HLS ARRAY_PARTITION variable = W1_i cyclic factor=32 dim = 2
 #pragma HLS ARRAY_PARTITION variable = W1_f cyclic factor=32 dim = 2
 #pragma HLS ARRAY_PARTITION variable = W1_c cyclic factor=32 dim = 2
 #pragma HLS ARRAY_PARTITION variable = W1_o cyclic factor=32 dim = 2

 #pragma HLS ARRAY_PARTITION variable = U1_i cyclic factor=32 dim = 2
 #pragma HLS ARRAY_PARTITION variable = U1_f cyclic factor=32 dim = 2
 #pragma HLS ARRAY_PARTITION variable = U1_c cyclic factor=32 dim = 2
 #pragma HLS ARRAY_PARTITION variable = U1_o cyclic factor=32 dim = 2

 #pragma HLS ARRAY_PARTITION variable = b1_i cyclic factor=32 dim = 1
 #pragma HLS ARRAY_PARTITION variable = b1_f cyclic factor=32 dim = 1
 #pragma HLS ARRAY_PARTITION variable = b1_c cyclic factor=32 dim = 1
 #pragma HLS ARRAY_PARTITION variable = b1_o cyclic factor=32 dim = 1

    static data_t h0_state[N_STATES] = {0};
    static data_t h1_oldstate[N_STATES] = {0};
    static data_t c1_oldstate[N_STATES] = {0};
    static data_t h1_newstate[N_STATES] = {0};
    static data_t c1_newstate[N_STATES] = {0};

#pragma HLS ARRAY_PARTITION variable = h0_state cyclic factor=64
#pragma HLS ARRAY_PARTITION variable = h1_oldstate cyclic factor=64
#pragma HLS ARRAY_PARTITION variable = c1_oldstate cyclic factor=64
#pragma HLS ARRAY_PARTITION variable = h1_newstate cyclic factor=64
#pragma HLS ARRAY_PARTITION variable = c1_newstate cyclic factor=64

    static int timestep = 0;

    fic::decoder<data_t, packet_config>(h0_in, h0_state);
    nn::lstm_static<data_t, config1, cell_act_config, recurrent_act_config>(h0_state, h1_oldstate,h1_newstate, c1_oldstate,c1_newstate, W1_i,W1_f,W1_c,W1_o,
        U1_i,U1_f,U1_c,U1_o, b1_i, b1_f, b1_c, b1_o);
    timestep++;
    fic::encoder<data_t, packet_config>(h1_newstate, h1_out);

    // reset memory between calls
    if (timestep == N_LOOP)
    {
//         for (int i=0; i<N_STATES; i++){
// #pragma HLS unroll
//     		h1_out[i] = h1_newstate[i];
//     	}

        for (int i = 0; i < N_STATES; i++)
        {
        #pragma HLS unroll
            h1_oldstate[i] = 0;
            c1_oldstate[i] = 0;
            h1_newstate[i] = 0;
            c1_newstate[i] = 0;
        }
        timestep = 0;
    }
}
