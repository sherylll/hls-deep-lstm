#include "parameter.h"
#include "../hlslib/fic_utils/fic_packet.h"

#include "weights/lstm0/Wi.h"
#include "weights/lstm0/Wf.h"
#include "weights/lstm0/Wc.h"
#include "weights/lstm0/Wo.h"
#include "weights/lstm0/Ui.h"
#include "weights/lstm0/Uf.h"
#include "weights/lstm0/Uc.h"
#include "weights/lstm0/Uo.h"
#include "weights/lstm0/bi.h"
#include "weights/lstm0/bf.h"
#include "weights/lstm0/bc.h"
#include "weights/lstm0/bo.h"
#include "weights/lstm0/proj.h"

#include "../test_data/mfcc_4.h"
void lstm0(ap_uint<4> rpi_ctrl,
		ap_uint<169> h0_packets[N_PACKETS])
{
#pragma HLS INTERFACE axis port = rpi_ctrl
#pragma HLS INTERFACE axis port = h0_packets

#pragma HLS ARRAY_PARTITION variable=Ui,Uf,Uc,Uo,Wi,Wf,Wc,Wo,proj complete dim = 2
#pragma HLS ARRAY_PARTITION variable=bi,bf,bc,bo cyclic factor=64

    static lstm_t h0_oldstate[N_PROJ] = {0};
    static lstm_t c0_oldstate[N_STATES] = {0};
     lstm_t h0_newstate[N_PROJ] = {0};
     lstm_t c0_newstate[N_STATES] = {0};
#pragma HLS ARRAY_PARTITION variable=h0_oldstate,c0_oldstate,h0_newstate,c0_newstate complete

    data_t mfcc[N_INPUTS];
#pragma HLS ARRAY_PARTITION variable = mfcc complete

    static short timestep = 0;
    ap_uint<4> ctrl = rpi_ctrl;
    switch (ctrl){
        case 0xf:
            for (int i = 0; i < N_INPUTS; i++)
#pragma HLS pipeline
                mfcc[i] = mfcc_inputs[timestep * N_INPUTS + i];

            nn::lstmp<data_t, lstm_t, config0, cell_act_config, recurrent_act_config>(mfcc, h0_oldstate, h0_newstate, c0_oldstate, c0_newstate,
                        Wi, Wf, Wc, Wo, Ui, Uf, Uc, Uo, proj, bi, bf, bc, bo);
            timestep++;
            fic::encoder<lstm_t, packet_config>(h0_newstate, h0_packets);
            break;
        case 0xe:
            for (int i = 0; i < N_PROJ; i++)
            {
#pragma HLS unroll
                h0_oldstate[i] = 0; h0_newstate[i] = 0;
            }
            for (int i = 0; i < N_STATES; i++)
            {
#pragma HLS unroll
                c0_oldstate[i] = 0; c0_newstate[i] = 0;
            }
            timestep = 0;
            break;
    }
}

/* 
This implementation using data from RPi does not work properly: the valid signal of rpi_in fails to turn on after the first run...

#define RPI_DATA_WIDTH 4
#define APP_DATA_WIDTH 32
#define NUM_DATA_BLOCKS APP_DATA_WIDTH / RPI_DATA_WIDTH

union float_and_uint {
    float fval;
    int ival;
};

void lstm0(ap_uint<RPI_DATA_WIDTH> rpi_in[NUM_DATA_BLOCKS * N_INPUTS],
		ap_uint<169> h0_packets[N_PACKETS])
{
#pragma HLS INTERFACE axis port = rpi_in
#pragma HLS INTERFACE axis port = h0_packets

#pragma HLS ARRAY_PARTITION variable=Ui,Uf,Uc,Uo,Wi,Wf,Wc,Wo,proj complete dim = 2
#pragma HLS ARRAY_PARTITION variable=bi,bf,bc,bo cyclic factor=64

    static lstm_t h0_oldstate[N_PROJ] = {0};
    static lstm_t c0_oldstate[N_STATES] = {0};

     lstm_t h0_newstate[N_PROJ] = {0};
     lstm_t c0_newstate[N_STATES] = {0};
#pragma HLS ARRAY_PARTITION variable=h0_oldstate,c0_oldstate,h0_newstate,c0_newstate complete

    data_t _data[N_INPUTS];
#pragma HLS ARRAY_PARTITION variable = _data complete
    
    // convert RPi input to 16-bit numbers
    for (int i = 0; i < N_INPUTS; i++){
#pragma HLS pipeline
    	int ival = 0;
    	union float_and_uint bits_to_float;
    	for (int j = 0; j < NUM_DATA_BLOCKS; j++)
    	{
    		int shifted = rpi_in[j + i * NUM_DATA_BLOCKS];
    		ival |= shifted << (APP_DATA_WIDTH - (j+1) * RPI_DATA_WIDTH);
    	}
    	bits_to_float.ival = ival;
    	_data[i] = (data_t) (bits_to_float.fval);
    }

    static short timestep = 0;

    nn::lstmp<data_t, lstm_t, config0, cell_act_config, recurrent_act_config>(_data, h0_oldstate, h0_newstate, c0_oldstate, c0_newstate,
				Wi, Wf, Wc, Wo, Ui, Uf, Uc, Uo, proj, bi, bf, bc, bo);
    timestep++;
    fic::encoder<lstm_t, packet_config>(h0_newstate, h0_packets);

    if (timestep == N_LOOP)
    {
        for (int i = 0; i < N_PROJ; i++)
        {
    #pragma HLS unroll
            h0_oldstate[i] = 0; h0_newstate[i] = 0;
        }
        for (int i = 0; i < N_STATES; i++)
        {
    #pragma HLS unroll
            c0_oldstate[i] = 0; c0_newstate[i] = 0;
        }
        timestep = 0;
    }
} */
