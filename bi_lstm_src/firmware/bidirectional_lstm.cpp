/*

4:  9
*/

#include "parameters.h"
#include "lstm.h"
#include "../../hlslib/fic_utils/fic_packet.h"
#include "../../test_data/test_digit_4.h"

struct recurrent_act_config
{
	typedef data_t table_t;
	static const unsigned n_in = N_STATES * 3;
	static const unsigned table_size = 4096;
	static const unsigned activation_type = nn::activ_hard_sigmoid; // Keras default
	static const unsigned unroll_factor = 64; // for unrolling hardsigmoid
};

void bidirectional_lstm(ap_uint<4> rx_start,
               data_t res[N_OUTPUTS])
{
#pragma HLS INTERFACE axis port = rx_start
#pragma HLS INTERFACE axis port = res
#pragma HLS ARRAY_RESHAPE variable = data cyclic factor=8 dim = 2
#pragma HLS ARRAY_RESHAPE variable = res complete dim = 1

    data_t Why[N_OUTPUTS][2*N_STATES] = WHY;
    data_t by[N_OUTPUTS] = BY;

    data_t W_i[N_STATES][N_INPUTS]=W_I, W_f[N_STATES][N_INPUTS]=W_F, W_c[N_STATES][N_INPUTS]=W_C, W_o[N_STATES][N_INPUTS]=W_O;
    data_t U_i[N_STATES][N_STATES]=U_I, U_f[N_STATES][N_STATES]=U_F, U_c[N_STATES][N_STATES]=U_C, U_o[N_STATES][N_STATES]=U_O;
    data_t b_i[N_STATES]=B_I, b_f[N_STATES]=B_F, b_c[N_STATES]=B_C, b_o[N_STATES]=B_O;

    data_t W1_i[N_STATES][N_STATES]=W1_I, W1_f[N_STATES][N_STATES]=W1_F, W1_c[N_STATES][N_STATES]=W1_C, W1_o[N_STATES][N_STATES]=W1_O;
    data_t U1_i[N_STATES][N_STATES]=U1_I, U1_f[N_STATES][N_STATES]=U1_F, U1_c[N_STATES][N_STATES]=U1_C, U1_o[N_STATES][N_STATES]=U1_O;
    data_t b1_i[N_STATES]=B1_I, b1_f[N_STATES]=B1_F, b1_c[N_STATES]=B1_C, b1_o[N_STATES]=B1_O;

 #pragma HLS ARRAY_PARTITION variable = W_i complete dim = 2
 #pragma HLS ARRAY_PARTITION variable = W_f complete dim = 2
 #pragma HLS ARRAY_PARTITION variable = W_c complete dim = 2
 #pragma HLS ARRAY_PARTITION variable = W_o complete dim = 2

 #pragma HLS ARRAY_PARTITION variable = U_i complete dim = 2
 #pragma HLS ARRAY_PARTITION variable = U_f complete dim = 2
 #pragma HLS ARRAY_PARTITION variable = U_c complete dim = 2
 #pragma HLS ARRAY_PARTITION variable = U_o complete dim = 2

 #pragma HLS ARRAY_PARTITION variable = b_i complete dim = 1
 #pragma HLS ARRAY_PARTITION variable = b_f complete dim = 1
 #pragma HLS ARRAY_PARTITION variable = b_c complete dim = 1
 #pragma HLS ARRAY_PARTITION variable = b_o complete dim = 1

#pragma HLS ARRAY_PARTITION variable = Why complete dim = 2
 #pragma HLS ARRAY_PARTITION variable = by complete

    static data_t h0_oldstate[N_STATES] = {0};
    static data_t c0_oldstate[N_STATES] = {0};
    static data_t h1_oldstate[N_STATES] = {0};
    static data_t c1_oldstate[N_STATES] = {0};

    static data_t h0_newstate[N_STATES] = {0};
    static data_t c0_newstate[N_STATES] = {0};
    static data_t h1_newstate[N_STATES] = {0};
    static data_t c1_newstate[N_STATES] = {0};

 #pragma HLS ARRAY_PARTITION variable = h0_state complete
 #pragma HLS ARRAY_PARTITION variable = c0_state complete
#pragma HLS ARRAY_PARTITION variable = h1_state complete
#pragma HLS ARRAY_PARTITION variable = c1_state complete

    data_t test_row[N_INPUTS];
    data_t test_row_rev[N_INPUTS];
#pragma HLS ARRAY_PARTITION variable = test_row complete
#pragma HLS ARRAY_PARTITION variable=image_digit_4 complete dim=2

    ap_uint<169> packets_test[16];
    for (int iloop = 0; iloop < N_LOOP; iloop++)
    {
        for (int i = 0; i < N_INPUTS; i++){
            test_row[i] = image_digit_4[iloop][i];
            test_row_rev[N_INPUTS-i] = test_row[i];
        }

        nn::lstm_static<data_t, config0, cell_act_config, recurrent_act_config>(test_row, h0_oldstate, h0_newstate, c0_oldstate, c0_newstate, W_i,W_f,W_c,W_o,
        		U_i,U_f,U_c,U_o, b_i, b_f, b_c, b_o);
        // fic::encoder<data_t, packet_config>(h0_newstate, packets_test);
        // fic::decoder<data_t, packet_config>(packets_test, h0_newstate);
        nn::lstm_static<data_t, config1, cell_act_config, recurrent_act_config>(test_row_rev, h1_oldstate,h1_newstate, c1_oldstate,c1_newstate, W1_i,W1_f,W1_c,W1_o,
        		U1_i,U1_f,U1_c,U1_o, b1_i, b1_f, b1_c, b1_o);
    }

     std::cout << "h1 "; for (int ii = 0; ii < N_STATES; ii++) std::cout << h1_newstate[ii]<< " "; std::cout << std::endl;

    data_t y[N_OUTPUTS] = {0};
#pragma HLS ARRAY_PARTITION variable = y complete
    data_t h_concat[N_STATES*2];
    for (int i=0; i<N_STATES; i++)
    {
        h_concat[i] = h0_newstate[i];
        h_concat[i+N_STATES] = h1_newstate[i];
    }
    nn::fc<data_t, config1::n_state * 2, config1::n_out>(Why, h_concat, by, y);
    nn::softmax<data_t, data_t, softmax_config>(y, res);

    // reset memory between calls
    for (int i = 0; i < N_STATES; i++)
    {
#pragma HLS unroll
        h0_oldstate[i] = 0;
        c0_oldstate[i] = 0;
        h1_oldstate[i] = 0;
        c1_oldstate[i] = 0;
    	h0_newstate[i] = 0;
        c0_newstate[i] = 0;
        h1_newstate[i] = 0;
        c1_newstate[i] = 0;
    }
}
