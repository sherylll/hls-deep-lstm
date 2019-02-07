#include "parameters.h"
#include "lstm.h"
#include "../../hlslib/fic_utils/fic_packet.h"

void lstm2(ap_uint<169> h1_in[N_PACKETS],
            ap_uint<169> h2_out[N_PACKETS])
            // data_t h2_out[N_STATES])
{
#pragma HLS INTERFACE axis port = h1_in
#pragma HLS INTERFACE axis port = h2_out
#pragma HLS dataflow

    data_t W2_i[N_STATES][N_STATES]=W2_I, W2_f[N_STATES][N_STATES]=W2_F, W2_c[N_STATES][N_STATES]=W2_C, W2_o[N_STATES][N_STATES]=W2_O;
    data_t U2_i[N_STATES][N_STATES]=U2_I, U2_f[N_STATES][N_STATES]=U2_F, U2_c[N_STATES][N_STATES]=U2_C, U2_o[N_STATES][N_STATES]=U2_O;
    data_t b2_i[N_STATES]=B2_I, b2_f[N_STATES]=B2_F, b2_c[N_STATES]=B2_C, b2_o[N_STATES]=B2_O;

 #pragma HLS ARRAY_PARTITION variable = W2_i cyclic factor=32 dim = 2
 #pragma HLS ARRAY_PARTITION variable = W2_f cyclic factor=32 dim = 2
 #pragma HLS ARRAY_PARTITION variable = W2_c cyclic factor=32 dim = 2
 #pragma HLS ARRAY_PARTITION variable = W2_o cyclic factor=32 dim = 2

 #pragma HLS ARRAY_PARTITION variable = U2_i cyclic factor=32 dim = 2
 #pragma HLS ARRAY_PARTITION variable = U2_f cyclic factor=32 dim = 2
 #pragma HLS ARRAY_PARTITION variable = U2_c cyclic factor=32 dim = 2
 #pragma HLS ARRAY_PARTITION variable = U2_o cyclic factor=32 dim = 2

 #pragma HLS ARRAY_PARTITION variable = b2_i cyclic factor=32 dim = 1
 #pragma HLS ARRAY_PARTITION variable = b2_f cyclic factor=32 dim = 1
 #pragma HLS ARRAY_PARTITION variable = b2_c cyclic factor=32 dim = 1
 #pragma HLS ARRAY_PARTITION variable = b2_o cyclic factor=32 dim = 1

    static data_t h1_state[N_STATES] = {0};
    static data_t h2_oldstate[N_STATES] = {0};
    static data_t c2_oldstate[N_STATES] = {0};
    static data_t h2_newstate[N_STATES] = {0};
    static data_t c2_newstate[N_STATES] = {0};

#pragma HLS ARRAY_PARTITION variable = h1_state cyclic factor=64
#pragma HLS ARRAY_PARTITION variable = h2_oldstate cyclic factor=64
#pragma HLS ARRAY_PARTITION variable = c2_oldstate cyclic factor=64
#pragma HLS ARRAY_PARTITION variable = h2_newstate cyclic factor=64
#pragma HLS ARRAY_PARTITION variable = c2_newstate cyclic factor=64

    static int timestep = 0;

    fic::decoder<data_t, packet_config>(h1_in, h1_state);
    nn::lstm_static<data_t, config1, cell_act_config, recurrent_act_config>(h1_state, h2_oldstate,h2_newstate, c2_oldstate,c2_newstate, W2_i,W2_f,W2_c,W2_o,
        U2_i,U2_f,U2_c,U2_o, b2_i, b2_f, b2_c, b2_o);
    timestep++;
    fic::encoder<data_t, packet_config>(h2_newstate, h2_out);

    // reset memory between calls
    if (timestep == N_LOOP)
    {
//         for (int i=0; i<N_STATES; i++){
// #pragma HLS unroll
//     		h2_out[i] = h2_newstate[i];
//     	}

        for (int i = 0; i < N_STATES; i++)
        {
        #pragma HLS unroll
            h2_oldstate[i] = 0;
            c2_oldstate[i] = 0;
            h2_newstate[i] = 0;
            c2_newstate[i] = 0;
        }
        timestep = 0;
    }
}
