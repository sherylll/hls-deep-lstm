// TODO add encoder/decoder for switch
#include "seq2seq.h"
#include "decoder_weights/Wi.h"
#include "decoder_weights/Wf.h"
#include "decoder_weights/Wc.h"
#include "decoder_weights/Wo.h"
#include "decoder_weights/Ui.h"
#include "decoder_weights/Uf.h"
#include "decoder_weights/Uc.h"
#include "decoder_weights/Uo.h"
#include "decoder_weights/bi.h"
#include "decoder_weights/bf.h"
#include "decoder_weights/bc.h"
#include "decoder_weights/bo.h"

#define START_ID 5

void decoder_lstm(lstm_t h_last[N_STATES], lstm_t c_last[N_STATES], int predicted_id[1], bool eos)
{
    // h_last and c_last come from the encoder FPGA, input come from IO
#pragma HLS interface axis port = input,predicted_id,h_last,c_last
#pragma HLS array_reshape variable = Ui,Uf,Wc,Wo,Wi,Wf,Wc,Wo dim = 2

    static bool started = false; 
    static lstm_t h_oldstate[N_STATES];
    static lstm_t c_oldstate[N_STATES];
// #pragma HLS ARRAY_PARTITION variable = h_oldstate
    lstm_t h_newstate[N_STATES];
    lstm_t c_newstate[N_STATES];

    static int cur_id;
    data_t input[N_EMBEDDING]; // lookup
    
    // initialize h and c with encoder h and c
    if (!started)
    {
        for (int i=0; i<N_STATES; i++)
        {
            h_oldstate[i] = h_last[i];
            c_oldstate[i] = c_last[i];
        }
        started = true;
        cur_id = START_ID;
    }

    // id to input vector
    for (int i=0; i<N_EMBEDDING; i++)
        input[i] = embedding[cur_id][i];

    nn::lstm<data_t, lstm_t, config_decoder, cell_act_config, recurrent_act_config>(input, h_oldstate, h_newstate, c_oldstate, c_newstate,
            Wi, Wf, Wc, Wo, Ui, Uf, Uc, Uo, bi, bf, bc, bo);

    for (int i=0; i<N_STATES; i++)
            c_oldstates[i] = c_newstate[i];

    ///////////// prediction //////////////////
    // TODO prune this part
    y_t y[N_OUTPUT_VOCAB];
    nn::fc<lstm_t, res_t, fc_config0>(Wy, h0_newstate, by, y);
    // TODO argmax (current impl. only for simulation)
    int max_id;
    y_t max_y = y[0];
    for (int i=0; i<N_OUTPUT_VOCAB; i++){
        if (y[i] > max_y){
            max_y = y[i];
            max_id = i;
        }
    }
    predicted_id[0] = max_id;
    cur_id = max_id;

    if (eos) // maybe eos could be coded into the header packet?
    {
        started = false;
        for (int i = 0; i < N_STATES; i++)
        {
            c_oldstate[i] = 0; 
            h_oldstate[i] = 0;
        }
    }
}