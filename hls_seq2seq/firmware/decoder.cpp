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

void decoder_lstm(lstm_t h_last[N_STATES], lstm_t c_last[N_STATES], data_t input[N_EMBEDDING], data_t decoder_h[N_STATES], bool eos)
{
    // input comes from another FiC and decoder_h is sent back to it
#pragma HLS interface axis port = input,output,h_last,c_last
#pragma HLS array_reshape variable = Ui,Uf,Wc,Wo,Wi,Wf,Wc,Wo dim = 2

    static lstm_t h_oldstate[N_STATES] = {0};
    static lstm_t c_oldstate[N_STATES] = {0};
#pragma HLS ARRAY_PARTITION variable = h_oldstate

    lstm_t h_newstate[N_STATES];
    lstm_t c_newstate[N_STATES];

    nn::lstm<data_t, lstm_t, config_decoder, cell_act_config, recurrent_act_config>(inputs, h_oldstate, h_newstate, c_oldstate, c_newstate,
            Wi, Wf, Wc, Wo, Ui, Uf, Uc, Uo, bi, bf, bc, bo);

    if (eos) // maybe eos could be coded into the header packet?
    {
        for (int i = 0; i < N_STATES; i++)
        {
            c_oldstate[i] = 0; 
            h_oldstate[i] = 0;
        }
        // send states
        for (int i = 0; i < N_STATES; i++)
        {
            h_last[i] = h_newstate[i];
            c_last[i] = c_newstate[i];
        }        
    }
}