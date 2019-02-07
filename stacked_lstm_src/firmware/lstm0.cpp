#include "parameters.h"
#include "lstm.h"
#include "../../hlslib/fic_utils/fic_packet.h"
#include "../../test_data/test_digit_4.h"

void lstm0(ap_uint<4> rx_start,
           ap_uint<169> h0_packets[N_PACKETS])
{
#pragma HLS INTERFACE axis port = rx_start
#pragma HLS INTERFACE axis port = h0_packets

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

    data_t test_row[N_INPUTS];
#pragma HLS ARRAY_PARTITION variable = test_row complete
#pragma HLS ARRAY_PARTITION variable=image_digit_4 complete dim=2

    static short timestep = 0;
    ap_uint<4> rx_start_tmp = rx_start;
    switch (rx_start_tmp){
    case 0xf:
        for (int i = 0; i < N_INPUTS; i++)
            test_row[i] = image_digit_4[timestep][i];

        nn::lstm_static<data_t, config0, cell_act_config, recurrent_act_config>(test_row, h0_oldstate, h0_newstate, c0_oldstate, c0_newstate, W_i,W_f,W_c,W_o,
        		U_i,U_f,U_c,U_o, b_i, b_f, b_c, b_o);
        timestep++;
        fic::encoder<data_t, packet_config>(h0_newstate, h0_packets);
    	break;
    case 0xe:
        for (int i = 0; i < N_STATES; i++)
        {
#pragma HLS unroll
            h0_oldstate[i] = 0;
            c0_oldstate[i] = 0;
            h0_newstate[i] = 0;
            c0_newstate[i] = 0;
        }
        timestep=0;
    	break;
    }
}
