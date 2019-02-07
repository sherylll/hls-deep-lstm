#include "parameters.h"
#include "lstm.h"
#include "../../hlslib/fic_utils/fic_packet.h"

void lstm3(ap_uint<169> h2_in[N_PACKETS],
            data_t h3_out[N_STATES])
{
#pragma HLS INTERFACE axis port = h2_in
#pragma HLS INTERFACE axis port = h3_out
#pragma HLS dataflow

    data_t W3_i[N_STATES][N_STATES]=W3_I, W3_f[N_STATES][N_STATES]=W3_F, W3_c[N_STATES][N_STATES]=W3_C, W3_o[N_STATES][N_STATES]=W3_O;
    data_t U3_i[N_STATES][N_STATES]=U3_I, U3_f[N_STATES][N_STATES]=U3_F, U3_c[N_STATES][N_STATES]=U3_C, U3_o[N_STATES][N_STATES]=U3_O;
    data_t b3_i[N_STATES]=B3_I, b3_f[N_STATES]=B3_F, b3_c[N_STATES]=B3_C, b3_o[N_STATES]=B3_O;

 #pragma HLS ARRAY_PARTITION variable = W3_i cyclic factor=32 dim = 2
 #pragma HLS ARRAY_PARTITION variable = W3_f cyclic factor=32 dim = 2
 #pragma HLS ARRAY_PARTITION variable = W3_c cyclic factor=32 dim = 2
 #pragma HLS ARRAY_PARTITION variable = W3_o cyclic factor=32 dim = 2

 #pragma HLS ARRAY_PARTITION variable = U3_i cyclic factor=32 dim = 2
 #pragma HLS ARRAY_PARTITION variable = U3_f cyclic factor=32 dim = 2
 #pragma HLS ARRAY_PARTITION variable = U3_c cyclic factor=32 dim = 2
 #pragma HLS ARRAY_PARTITION variable = U3_o cyclic factor=32 dim = 2

 #pragma HLS ARRAY_PARTITION variable = b3_i cyclic factor=32 dim = 1
 #pragma HLS ARRAY_PARTITION variable = b3_f cyclic factor=32 dim = 1
 #pragma HLS ARRAY_PARTITION variable = b3_c cyclic factor=32 dim = 1
 #pragma HLS ARRAY_PARTITION variable = b3_o cyclic factor=32 dim = 1

    static data_t h2_state[N_STATES] = {0};
    static data_t h3_oldstate[N_STATES] = {0};
    static data_t c3_oldstate[N_STATES] = {0};
    static data_t h3_newstate[N_STATES] = {0};
    static data_t c3_newstate[N_STATES] = {0};

#pragma HLS ARRAY_PARTITION variable = h2_state cyclic factor=64
#pragma HLS ARRAY_PARTITION variable = h3_oldstate cyclic factor=64
#pragma HLS ARRAY_PARTITION variable = c3_oldstate cyclic factor=64
#pragma HLS ARRAY_PARTITION variable = h3_newstate cyclic factor=64
#pragma HLS ARRAY_PARTITION variable = c3_newstate cyclic factor=64

    static int timestep = 0;

    fic::decoder<data_t, packet_config>(h2_in, h2_state);
    nn::lstm_static<data_t, config1, cell_act_config, recurrent_act_config>(h2_state, h3_oldstate,h3_newstate, c3_oldstate,c3_newstate, W3_i,W3_f,W3_c,W3_o,
        U3_i,U3_f,U3_c,U3_o, b3_i, b3_f, b3_c, b3_o);
    timestep++;

    // reset memory between calls
    if (timestep == N_LOOP)
    {
        for (int i=0; i<N_STATES; i++){
#pragma HLS unroll
    		h3_out[i] = h3_newstate[i];
    	}

        for (int i = 0; i < N_STATES; i++)
        {
        #pragma HLS unroll
            h3_oldstate[i] = 0;
            c3_oldstate[i] = 0;
            h3_newstate[i] = 0;
            c3_newstate[i] = 0;
        }
        timestep = 0;
    }
}
