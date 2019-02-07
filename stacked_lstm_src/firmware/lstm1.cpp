#include "parameters.h"
#include "lstm.h"
#include "../../hlslib/fic_utils/fic_packet.h"

struct recurrent_act_config
{
	typedef data_t table_t;
	static const unsigned n_in = N_STATES * 3;
	static const unsigned table_size = 4096;
	static const unsigned activation_type = nn::activ_hard_sigmoid; // Keras default
	static const unsigned unroll_factor = 32;
};

void lstm1(ap_uint<169> h0_in[N_PACKETS],
            data_t h1_out[N_STATES])
{
#pragma HLS INTERFACE axis port = h0_in
#pragma HLS INTERFACE axis port = h1_out
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

// #pragma HLS ARRAY_PARTITION variable = Why cyclic factor=64 dim = 2
//  #pragma HLS ARRAY_PARTITION variable = by complete

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
    nn::lstm_static<data_t, config1, cell_act_config, recurrent_act_config>(h0_state, h1_oldstate,h1_newstate, c1_oldstate,c1_newstate, W2_i,W2_f,W2_c,W2_o,
        U2_i,U2_f,U2_c,U2_o, b2_i, b2_f, b2_c, b2_o);
    timestep++;

//    data_t y[N_OUTPUTS] = {0};
//#pragma HLS ARRAY_PARTITION variable = y complete
//    nn::fc<data_t, config1::n_state, config1::n_out>(Why, h1_state, by, y);
//    nn::softmax<data_t, data_t, softmax_config>(y, res);

    // reset memory between calls
    if (timestep == N_LOOP)
    {
        for (int i=0; i<N_STATES; i++){
#pragma HLS unroll
    		h1_out[i] = h1_newstate[i];
    	}

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
