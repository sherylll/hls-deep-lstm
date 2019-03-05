/*
simulation does not work due to large array size; internal compiler error returned
converted from hidden_states_digit_4_stacked.txt
h3[27]: 522 1951 61829 64947 38 2091 61706 1650 1792 63128 3196 1587 62951 64103 2081 566 65275 2632 64275 62165 1161 65055 64603 156 3673 64460 64967 2105 757 65509 2973 2793 63878 1189 63288 451 62089 675 1369 63516 2727 64788 700 62263 63510 62297 2164 64077 65432 64591 62128 715 3415 63916 62223 65118 65457 376 4095 525 360 2233 2496 124 1405 63791 62380 380 343 64402 2118 1934 2426 701 3180 2807 62389 64771 64914 64330 1503 199 110 62041 1019 63502 797 64716 63802 2426 64046 3468 2257 65288 64083 64842 64007 62597 580 64889 65449 1197 64629 2135 63281 2216 3363 1150 2639 2046 63679 65334 64472 63287 1528 64559 2609 64184 828 65127 2102 2622 64323 1653 65320 65163 64983 65013 
*/

#include "parameters.h"
#include "lstm.h"
#include "../../hlslib/fic_utils/fic_packet.h"
#include "../../test_data/test_digit_4.h"

void stacked_lstm(ap_uint<4> rx_start,
                  data_t res[N_OUTPUTS])
{
#pragma HLS INTERFACE axis port = rx_start
#pragma HLS INTERFACE axis port = res
#pragma HLS ARRAY_RESHAPE variable = data cyclic factor = 8 dim = 2
#pragma HLS ARRAY_RESHAPE variable = res complete dim = 1

    data_t Why[N_OUTPUTS][N_STATES] = WHY;
    data_t by[N_OUTPUTS] = BY;

   data_t W_i[N_STATES][N_INPUTS] = W_I, W_f[N_STATES][N_INPUTS] = W_F, W_c[N_STATES][N_INPUTS] = W_C, W_o[N_STATES][N_INPUTS] = W_O;
   data_t U_i[N_STATES][N_STATES] = U_I, U_f[N_STATES][N_STATES] = U_F, U_c[N_STATES][N_STATES] = U_C, U_o[N_STATES][N_STATES] = U_O;
   data_t b_i[N_STATES] = B_I, b_f[N_STATES] = B_F, b_c[N_STATES] = B_C, b_o[N_STATES] = B_O;

   data_t W1_i[N_STATES][N_STATES] = W1_I, W1_f[N_STATES][N_STATES] = W1_F, W1_c[N_STATES][N_STATES] = W1_C, W1_o[N_STATES][N_STATES] = W1_O;
   data_t U1_i[N_STATES][N_STATES] = U1_I, U1_f[N_STATES][N_STATES] = U1_F, U1_c[N_STATES][N_STATES] = U1_C, U1_o[N_STATES][N_STATES] = U1_O;
   data_t b1_i[N_STATES] = B1_I, b1_f[N_STATES] = B1_F, b1_c[N_STATES] = B1_C, b1_o[N_STATES] = B1_O;

    data_t W2_i[N_STATES][N_STATES] = W2_I, W2_f[N_STATES][N_STATES] = W2_F, W2_c[N_STATES][N_STATES] = W2_C, W2_o[N_STATES][N_STATES] = W2_O;
    data_t U2_i[N_STATES][N_STATES] = U2_I, U2_f[N_STATES][N_STATES] = U2_F, U2_c[N_STATES][N_STATES] = U2_C, U2_o[N_STATES][N_STATES] = U2_O;
    data_t b2_i[N_STATES] = B2_I, b2_f[N_STATES] = B2_F, b2_c[N_STATES] = B2_C, b2_o[N_STATES] = B2_O;

   data_t W3_i[N_STATES][N_STATES] = W3_I, W3_f[N_STATES][N_STATES] = W3_F, W3_c[N_STATES][N_STATES] = W3_C, W3_o[N_STATES][N_STATES] = W3_O;
   data_t U3_i[N_STATES][N_STATES] = U3_I, U3_f[N_STATES][N_STATES] = U3_F, U3_c[N_STATES][N_STATES] = U3_C, U3_o[N_STATES][N_STATES] = U3_O;
   data_t b3_i[N_STATES] = B3_I, b3_f[N_STATES] = B3_F, b3_c[N_STATES] = B3_C, b3_o[N_STATES] = B3_O;

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
    static data_t h2_oldstate[N_STATES] = {0};
    static data_t c2_oldstate[N_STATES] = {0};
    static data_t h3_oldstate[N_STATES] = {0};
    static data_t c3_oldstate[N_STATES] = {0};

    static data_t h0_newstate[N_STATES] = {0};
    static data_t c0_newstate[N_STATES] = {0};
    static data_t h1_newstate[N_STATES] = {0};
    static data_t c1_newstate[N_STATES] = {0};
    static data_t h2_newstate[N_STATES] = {0};
    static data_t c2_newstate[N_STATES] = {0};
    static data_t h3_newstate[N_STATES] = {0};
    static data_t c3_newstate[N_STATES] = {0};

#pragma HLS ARRAY_PARTITION variable = h0_state complete
#pragma HLS ARRAY_PARTITION variable = c0_state complete
#pragma HLS ARRAY_PARTITION variable = h1_state complete
#pragma HLS ARRAY_PARTITION variable = c1_state complete

    data_t test_row[N_INPUTS];
#pragma HLS ARRAY_PARTITION variable = test_row complete
#pragma HLS ARRAY_PARTITION variable = image_digit_4 complete dim = 2

    ap_uint<169> packets_test[16];
    for (int iloop = 0; iloop < N_LOOP; iloop++)
    {
        for (int i = 0; i < N_INPUTS; i++)
        {
            test_row[i] = image_digit_4[iloop][i];
        }

       nn::lstm_static<data_t, config0, cell_act_config, recurrent_act_config>(test_row, h0_oldstate, h0_newstate, c0_oldstate, c0_newstate, W_i, W_f, W_c, W_o,
                                                                               U_i, U_f, U_c, U_o, b_i, b_f, b_c, b_o);
//        // fic::encoder<data_t, packet_config>(h0_newstate, packets_test);
//        // fic::decoder<data_t, packet_config>(packets_test, h0_newstate);
       nn::lstm_static<data_t, config1, cell_act_config, recurrent_act_config>(h0_newstate, h1_oldstate, h1_newstate, c1_oldstate, c1_newstate, W1_i, W1_f, W1_c, W1_o,
                                                                               U1_i, U1_f, U1_c, U1_o, b1_i, b1_f, b1_c, b1_o);

        nn::lstm_static<data_t, config1, cell_act_config, recurrent_act_config>(h1_newstate, h2_oldstate, h2_newstate, c2_oldstate, c2_newstate, W2_i, W2_f, W2_c, W2_o,
                                                                                U2_i, U2_f, U2_c, U2_o, b2_i, b2_f, b2_c, b2_o);

       nn::lstm_static<data_t, config1, cell_act_config, recurrent_act_config>(h2_newstate, h3_oldstate, h3_newstate, c3_oldstate, c3_newstate, W3_i, W3_f, W3_c, W3_o,
                                                                               U3_i, U3_f, U3_c, U3_o, b3_i, b3_f, b3_c, b3_o);
        // std::cout << "h0 ";
        // for (int ii = 0; ii < N_STATES; ii++)
        //     std::cout << h0_newstate[ii] << " ";
        // std::cout << std::endl;

        std::cout << "h3 ";
        for (int ii = 0; ii < N_STATES; ii++)
            std::cout << h3_newstate[ii] << " ";
        std::cout << std::endl;
    }

    data_t y[N_OUTPUTS] = {0};
    nn::fc<data_t, config1::n_state, config1::n_out>(Why, h3_newstate, by, y);
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
