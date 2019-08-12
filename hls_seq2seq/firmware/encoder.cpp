// TODO add encoder/decoder for switch
#include "seq2seq.h"

#include "encoder_weights/Wi.h"
#include "encoder_weights/Wf.h"
#include "encoder_weights/Wc.h"
#include "encoder_weights/Wo.h"
#include "encoder_weights/Ui.h"
#include "encoder_weights/Uf.h"
#include "encoder_weights/Uc.h"
#include "encoder_weights/Uo.h"
#include "encoder_weights/bi.h"
#include "encoder_weights/bf.h"
#include "encoder_weights/bc.h"
#include "encoder_weights/bo.h"

void encoder_lstm(data_t inputs[N_EMBEDDING], bool eos, lstm_t h_last[N_STATES], lstm_t c_last[N_STATES])
{
#pragma HLS interface axis port = inputs,h_last,c_last
#pragma HLS array_reshape variable = Ui,Uf,Wc,Wo,Wi,Wf,Wc,Wo dim = 2

    static lstm_t h_oldstate[N_STATES] = {0};
    static lstm_t c_oldstate[N_STATES] = {0};
#pragma HLS ARRAY_PARTITION variable = h_oldstate

    lstm_t h_newstate[N_STATES];
    lstm_t c_newstate[N_STATES];

    nn::lstm<data_t, lstm_t, config_encoder, cell_act_config, recurrent_act_config>(inputs, h_oldstate, h_newstate, c_oldstate, c_newstate,
            Wi, Wf, Wc, Wo, Ui, Uf, Uc, Uo, bi, bf, bc, bo);

    for (int i = 0; i < N_STATES; i++)
    {
        c0_oldstate[i] = c0_newstate[i];
        h0_oldstate[i] = h0_newstate[i];
    }

    if (eos) // for debugging: eos should be set to true when <end> is seen
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